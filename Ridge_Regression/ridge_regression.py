import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

class RidgeRegression:
    def __init__(self, lambda_=1):
        self.w = None
        self.lambda_ = lambda_

    def fit(self, x, y, axis=None):
        # 行列Xの左側に1を追加
        
        x = np.c_[np.ones(x.shape[0]),x]
        I = np.eye(x.shape[1])
        w = np.linalg.inv(x.T @ x + self.lambda_ * I) @ x.T @ y
        self.coef_ = w
    
    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]),x]
        return x @ self.coef_
