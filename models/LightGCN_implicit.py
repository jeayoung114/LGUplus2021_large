import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time
import numpy as np
import scipy.sparse as sp


class LightGCN_implicit(nn.Module):
    def __init__(self, train, valid, learning_rate, regs, batch_size, num_epochs, emb_size, num_layers, node_dropout, use_bpr, device):
        super(LightGCN_implicit, self).__init__()
        
        self.train_mat = sp.csr_matrix(train)
        self.valid_mat = sp.csr_matrix(valid)

        self.num_users, self.num_items = self.train_mat.shape

        self.R = sp.csr_matrix(train)

        self.norm_adj = self.create_adj_mat()

        self.learning_rate = learning_rate
        self.device = device
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.node_dropout = node_dropout
        self.use_bpr = use_bpr

        self.decay = regs

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.num_users,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.num_items,
                                                 self.emb_size)))
        })
        
        return embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):

        # 사용자-긍정 항목, 사용자-부정 항목 점수 계산
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        # BPR Loss 계산
        bpr_loss = -1 * torch.sum(nn.LogSigmoid()(pos_scores - neg_scores))

        # Weight L2 정규화
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer

        return bpr_loss + emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, pos_items, neg_items, drop_flag=False):
        # 사용자-항목 상호작용 그래프 정점 드롭아웃
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj
        # 초기 임베딩 불러오기
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        # 각각의 레이어에서의 임베딩 결과를 저장하기 위한 공간 생성
        all_embeddings = [ego_embeddings]

        # 레이어마다 GCN 수행
        for k in range(self.num_layers):
            # 메시지 통합 (Message Aggregation)
            norm_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # k번째 임베딩 저장
            all_embeddings += [norm_embeddings]
        
        # 동일 가중치 합으로 최종 임베딩 생성
        all_embeddings = torch.stack(all_embeddings, 1)
        final_embeddings = torch.mean(all_embeddings, 1)

        u_g_embeddings = final_embeddings[:self.num_users, :]
        i_g_embeddings = final_embeddings[self.num_users:, :]

        # 필요한 임베딩 가져가기
        u_g_embeddings = u_g_embeddings[users, :]
        pos_i_g_embeddings = i_g_embeddings[pos_items, :]
        neg_i_g_embeddings = i_g_embeddings[neg_items, :]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, i_g_embeddings

    def fit(self):
        user_idx = np.arange(self.num_users)

        for epoch in range(self.num_epochs):
            start_time = time()

            epoch_loss = 0.0

            self.train()

            np.random.RandomState(12345).shuffle(user_idx)

            batch_num = int(len(user_idx) / self.batch_size) + 1

            if self.use_bpr:
                data_loader = PairwiseGenerator(self.train_mat.toarray(), num_negatives=1, batch_size=self.batch_size, shuffle=True, device=self.device)

                for batch_data in data_loader:
                    user, pos_items, neg_items = batch_data

                    batch_loss = self.train_model_per_batch(user, pos_items, neg_items)

                    if torch.isnan(batch_loss):
                        print('Loss NAN. Train finish.')
                        break

                    epoch_loss += batch_loss

            else:
                for batch_idx in range(batch_num):
                    batch_users = user_idx[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                    batch_matrix = torch.FloatTensor(self.train_mat[batch_users, :].toarray()).to(self.device)
                    batch_users = torch.LongTensor(batch_users).to(self.device)
                    batch_loss = self.train_model_per_batch(batch_matrix, batch_users)

                    if torch.isnan(batch_loss):
                        print('Loss NAN. Train finish.')
                        break
                    
                    epoch_loss += batch_loss
            
            print('Epoch %d [%.1fs]: train_loss [%.5f = %.5f + %.5f]' % (epoch, time() - start_time, epoch_loss, 0, 0))

    def train_model_per_batch(self, train_matrix, batch_users, pos_items=0, neg_items=0):
        # grad 초기화
        self.optimizer.zero_grad()

        if self.use_bpr:
            u_embeddings, pos_i_embeddings, neg_i_embeddings, _ = self.forward(batch_users, pos_items, neg_items)

            # loss 구함
            loss = self.create_bpr_loss(u_embeddings, pos_i_embeddings, neg_i_embeddings)

        else:
            u_g_embeddings, _, _, i_g_embeddings = self.forward(batch_users, 0, 0)

            output = torch.sigmoid(torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)))

            # loss 구함
            loss = F.binary_cross_entropy(output, train_matrix, reduction='none').sum(1).mean()

        # 역전파
        loss.backward()

        # 최적화
        self.optimizer.step()

        return loss


    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            u_g_embeddings, _, _, i_g_embeddings = self.forward(user_ids, 0, 0)
            output = torch.sigmoid(torch.matmul(u_g_embeddings, torch.transpose(i_g_embeddings, 0, 1)))
            predict_ = output.detach().cpu().numpy()
            return predict_[item_ids]


    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)

        adj_mat = adj_mat.tolil()
        R = sp.csr_matrix(self.R).tolil()

        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()

        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj_mat.sum(axis=1))

        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        norm_adj = norm_adj.tocsr()

        return norm_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class PairwiseGenerator:
    def __init__(self, input_matrix, num_negatives=1, batch_size=32, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.num_negs = num_negatives

        self.num_users, self.num_items = input_matrix.shape
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.pos_dict = {}
        self.neg_dict = {}

        self._construct()

    def _construct(self):
        for u in range(self.num_users):
            u_items = self.input_matrix[u]
            self.pos_dict[u] = u_items
        pos_items = np.where(self.input_matrix[u] > 0.5)[0]
        for u in range(self.num_users):
            neg_items = list(set(range(self.num_items)) - set(pos_items))
            self.neg_dict[u] = neg_items

    def __len__(self):
        return int(np.ceil(self.num_users / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.num_users) 
        else:
            perm = np.arange(self.num_users)

        for b, st in enumerate(range(0, len(perm), self.batch_size)):
            batch_pos = []
            batch_neg = []

            ed = min(st + self.batch_size, len(perm))
            batch_users = perm[st:ed]

            for i, u in enumerate(batch_users):
                posForUser = self.pos_dict[u]
                negForUser = self.neg_dict[u]

                if len(posForUser) == 0:
                    continue

                posindex = np.random.randint(0, len(posForUser), size=1)[0]
                positem = posForUser[posindex]

                negindex = np.random.randint(0, len(negForUser), size=1)[0]
                negitem = negForUser[negindex]

                batch_pos.append(positem)
                batch_neg.append(negitem)

            batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
            batch_neg = torch.tensor(batch_neg, dtype=torch.long, device=self.device)

            yield batch_users, batch_pos, batch_neg