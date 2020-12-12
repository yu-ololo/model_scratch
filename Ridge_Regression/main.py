import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import ridge_regression

if __name__ == '__main__':
    boston = load_boston()
    # 説明変数：13項目
    # x = np.array([np.concatenate(([1], v)) for v in boston.data])
    x = boston.data
    # 目的変数：住宅価格のデータを使う
    y = boston.target
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    #リッジ回帰
    model = ridge_regression.RidgeRegression()
    model.fit(train_x, train_y)
    y = model.predict(test_x)

    for i in range(5):
        print("{:1.0f} {:5.3f}".format(test_y[i], y[i]))
    print("RMSE", np.sqrt(((test_y -y)**2).mean()))
    

