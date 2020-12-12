import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

class LinerRegression:
    def __init__(self):
        self.w = None

    def fit(self, x, y, axis=None):
        # 行列Xの左側に1を追加
        x = np.c_[np.ones(x.shape[0]),x]
        # x_mean = x.mean(axis=axis, keepdims=True)
        # x_std  = np.std(x, axis=axis, keepdims=True)
        # x = (x - x_mean) / x_std
        w = np.linalg.inv(x.T @ x) @ x.T @ y
        self.coef_ = w
    
    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]),x]
        return x @ self.coef_
