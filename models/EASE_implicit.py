"""
Embarrassingly shallow autoencoders for sparse data, 
Harald Steck,
Arxiv.
"""
import os
import math
import numpy as np

class EASE_implicit():
    def __init__(self, train, valid, reg_lambda):
        self.train = train.toarray()
        self.valid = valid.toarray()
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.reg_lambda = reg_lambda

    def fit(self):   
        train_matrix = self.train
        G = train_matrix.T @ train_matrix
        diag = np.diag_indices(G.shape[0]) # 대각 원소 인덱스 반환
        G[diag] += self.reg_lambda # 대각 원소 값 부분에 정규화 값 더해줌
        P = np.linalg.inv(G) # P_hat 행렬 계산
        
        # P_hat 행렬을 이용해 W 행렬 계산
        self.enc_w = P / (-np.diag(P)) 
        self.enc_w[diag] = 0

        # 사용자-항목 행렬과 W 행렬의 행렬 곱을 통해 예측 값 행렬 생성
        self.reconstructed = self.train @ self.enc_w

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]


