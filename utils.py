import os
import math
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google_drive_downloader import GoogleDriveDownloader as gdd


# 2d array를 dictionary로 만듦
# input: [[user_id, item_id, timestamp], ...] 형태의 numpy array
# output: {user_id: [item1, item2, ......], ...} 형태의 dictionary
def make_to_dict(data):
    data_dict = {}
    cur_user = -1
    tmp_user = []
    for row in data:
        user_id, item_id = row[0], row[1]
        if user_id != cur_user:
            if cur_user != -1:
                tmp = np.asarray(tmp_user)
                tmp_items = tmp[:,1]
                data_dict[cur_user] = list(tmp_items)
                
            cur_user = user_id
            tmp_user = []
        tmp_user.append(row)

    if cur_user != -1:
        tmp = np.asarray(tmp_user)
        tmp_items = tmp[:,1]
        data_dict[cur_user] = list(tmp_items)
        
    return data_dict


"""
dataset 관련 함수
"""
def load_data(data_name, implicit=True):
    data_path = './data/%s'%(data_name)
    if not os.path.exists(data_path):
        if 'small' in data_name:
            # https://drive.google.com/file/d/1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1_HFBNRk-FUOO1nquQVfWbD1IiqnWNzOW',
                dest_path=data_path,
            )
        elif '2m' in data_name:
            # https://drive.google.com/file/d/1f2bzvZw87Gu2yMpNm6EnxE6uHJYbbBUn/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='1f2bzvZw87Gu2yMpNm6EnxE6uHJYbbBUn',
                dest_path=data_path,
            )
        else: # 5m
            # https://drive.google.com/file/d/19ft9n-gO3rJYKmkVyejD3H94IHGlQVzU/view?usp=sharing
            gdd.download_file_from_google_drive(
                file_id='19ft9n-gO3rJYKmkVyejD3H94IHGlQVzU',
                dest_path=data_path,
            )
        print("데이터 다운로드 완료!")

    # 데이터셋 불러오기
    column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'people', 'country', 'genre']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        # print("Binarize")
        # movie_data = movie_data[movie_data['ratings'] >= 4]
        movie_data['rating'] = 1

    # 전체 데이터셋의 user, item 수 확인
    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid : new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in  movie_data.user_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    # movie title와 id를 매핑할 dict를 생성
    idx2title = {}
    if 'small' in data_name:
        item_id_title = movie_data[['item_id', 'title']]
        item_id_title = item_id_title.drop_duplicates() 
        for i, c in zip(item_id_title['item_id'], item_id_title['title']): 
            idx2title[i] = c

    # user 별 train, valid, test 나누기
    movie_data = movie_data[['user_id', 'item_id', 'rating']]
    movie_data = movie_data.sort_values(by="user_id", ascending=True)

    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify = movie_data['user_id'], random_state = 1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify = train_valid['user_id'], random_state = 1234)

    train = train.to_numpy()
    valid = valid.to_numpy()
    test = test.to_numpy()

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in train:
        matrix[u, i] = r
    train = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in valid:
        matrix[u, i] = r
    valid = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in test:
        matrix[u, i] = r
    test = sparse.csr_matrix(matrix)

    return train, valid, test, idx2title


# Precision, Recall, NDCG@K 평가
# input
#    - pred_u: 예측 값으로 정렬 된 item index
#    - target_u: test set의 item index
#    - top_k: top-k에서의 k 값
# output: prec_k, recall_k, ndcg_k의 점수
def compute_metrics(pred_u, target_u, top_k):
    pred_k = pred_u[:top_k]
    num_target_items = len(target_u)

    hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    num_hits = len(hits_k)

    idcg_k = 0.0
    for i in range(1, min(num_target_items, top_k) + 1):
        idcg_k += 1 / math.log(i + 1, 2)

    dcg_k = 0.0
    for idx, item in hits_k:
        dcg_k += 1 / math.log(idx + 1, 2)
    
    prec_k = num_hits / top_k
    recall_k = num_hits / min(num_target_items, top_k)
    ndcg_k = dcg_k / idcg_k

    return prec_k, recall_k, ndcg_k


def eval_implicit(model, train_data, test_data, top_k):
    prec_list = []
    recall_list = []
    ndcg_list = []
    
    if 'Item' in model.__class__.__name__:
        train_data = train_data.toarray()
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = train_data[:,item_id]
            missing_user_ids = np.where(train_by_item == 0)[0] # missing user_id

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            train_by_user = train_data[user_id]
            missing_item_ids = np.where(train_by_user == 0)[0] # missing item_id

            pred_u_score = pred_matrix[user_id, missing_item_ids]
            pred_u_idx = np.argsort(pred_u_score)[::-1]
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id].toarray()
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
    else:
        for user_id in range(train_data.shape[0]):
            train_by_user = train_data[user_id].toarray()[0]
            # print(train_by_user[0])
            missing_item_ids = np.where(train_by_user == 0)[0] # missing item_id

            pred_u_score = model.predict(user_id, missing_item_ids)
            pred_u_idx = np.argsort(pred_u_score)[::-1] # 내림차순 정렬
            pred_u = missing_item_ids[pred_u_idx]

            test_by_user = test_data[user_id].toarray()[0]
            # print(test_by_user[0])
            target_u = np.where(test_by_user >= 0.5)[0]

            prec_k, recall_k, ndcg_k = compute_metrics(pred_u, target_u, top_k)
            prec_list.append(prec_k)
            recall_list.append(recall_k)
            ndcg_list.append(ndcg_k)
    
    return np.mean(prec_list), np.mean(recall_list), np.mean(ndcg_list)

def eval_explicit(model, train_data, test_data):
    rmse_list = []
    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = test_data[:,item_id]
            missing_user_ids = np.where(train_by_item >= 0.5)[0]

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = pred_matrix[user_id, target_u]

            rmse = mean_squared_error(target_u_score, pred_u_score, squared=False)
            rmse_list.append(rmse)
    else:
        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = model.predict(user_id, target_u)

            # RMSE 계산
            rmse = mean_squared_error(target_u_score, pred_u_score, squared=False)
            rmse_list.append(rmse)

    return np.mean(rmse_list)
