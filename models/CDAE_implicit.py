"""
Collaborative Denoising Auto-Encoders for Top-N Recommender Systems, 
Yao Wu et al.,
WSDM 2016.
"""
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDAE_implicit(torch.nn.Module):
    def __init__(self, train, valid, num_epochs, hidden_dim, learning_rate, reg_lambda, dropout, device, activation="sigmoid", loss="CE", batch_size=1024):
        super().__init__()
        self.train_mat = train
        self.valid_mat = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.activation = activation
        self.loss_function = loss
        self.dropout = dropout

        self.device = device
        self.batch_size = batch_size

        self.build_graph()


    def build_graph(self):
        # W, W'와 b, b', V 만들기
        self.enc_w = nn.Parameter(torch.ones(self.num_items, self.hidden_dim))
        self.enc_b = nn.Parameter(torch.ones(self.hidden_dim))
        nn.init.xavier_uniform_(self.enc_w)
        nn.init.normal_(self.enc_b, 0, 0.001)

        self.dec_w = nn.Parameter(torch.ones(self.hidden_dim, self.num_items))
        self.dec_b = nn.Parameter(torch.ones(self.num_items))
        nn.init.xavier_uniform_(self.dec_w)
        nn.init.normal_(self.dec_b, 0, 0.001)

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)


    def forward(self, u, x):
        # 입력의 일부를 제거
        denoised_x = F.dropout(x, self.dropout, training=self.training) 

        # encoder 과정
        if self.activation == 'None':
            h = denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u)
        elif self.activation == 'tanh':
            h = torch.tanh(denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u))
        else:
            h = torch.sigmoid(denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u))

        # decoder 과정
        output = torch.sigmoid(h @ self.dec_w + self.dec_b)
        return output


    def fit(self):
        user_idx = np.arange(self.num_users)
        user_idx_torch = torch.LongTensor(user_idx).to(self.device)
        for epoch in range(0, self.num_epochs):
            start = time.time()
            epoch_loss = 0
            self.train()

            np.random.RandomState(12345).shuffle(user_idx)
            batch_num = int(len(user_idx) / self.batch_size) +1
            for batch_idx in range(batch_num):    
                user_idx_torch = torch.LongTensor(user_idx[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]).to(self.device)

                batch_matrix = torch.FloatTensor(self.train_mat[user_idx[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size], :].toarray()).to(self.device)
                batch_loss = self.train_model_per_batch(user_idx_torch, batch_matrix)
                if torch.isnan(batch_loss):
                    print('Loss NAN. Train finish.')
                    break
                epoch_loss += batch_loss
             
            if epoch % 10 == 0:
                print('epoch %d  loss: %.4f  training time per epoch:  %.2fs' % (epoch + 1, epoch_loss/batch_num, time.time() - start))
        print('final epoch %d  loss: %.4f  training time per epoch:  %.2fs' % (epoch + 1, epoch_loss/batch_num, time.time() - start))



    def train_model_per_batch(self, user_idx, train_matrix):
        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forwrad
        output = self.forward(user_idx, train_matrix)

        # loss 구함
        if self.loss_function == 'MSE':
            loss = F.mse_loss(output, train_matrix, reduction='none').sum(1).mean()
        else:
            loss = F.binary_cross_entropy(output, train_matrix, reduction='none').sum(1).mean()

        # 역전파
        loss.backward()

        # 최적화
        self.optimizer.step()
        return loss


    def predict(self, user_id, item_ids):
        self.eval()
        with torch.no_grad():
            user_idx_torch = torch.LongTensor([user_id]).to(self.device)
            eval_matrix = torch.FloatTensor(self.train_mat[user_id].toarray()).to(self.device)
            eval_output = self.forward(user_idx_torch, eval_matrix).detach().cpu().numpy()
        return eval_output[0, item_ids]
