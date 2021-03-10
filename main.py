#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv
import pickle
import datetime
import time
from sklearn.model_selection import train_test_split

from utils import *

from reg_alg import *

def train(X_train, X_dev, y_train, y_dev):
    print("Begin training...")
    
    # Linear regression:
    #model = linearRegression(X_train, X_dev, y_train, y_dev)

    # Ridge regression:

    print(RidgeRegression(X_train,y_train))

    # Lasso regression:

    # ElasticNet regression:

    # ExtraTree regression:

    # Decision Tree regression:

    # Random Forest regression:

    # Gradient Boosting Random Forest Regression

    print("End training...")

def main():
    args = args_parser()
    print("loading data...")
    cols_X, exposure, click_rate, cost, browse_rate, order_rate, roi = load_data()
    X_train, X_dev, y_train, y_dev = train_test_split(cols_X, exposure, test_size=0.1, random_state=42, shuffle=True) 
    print('X_train.shape:{0}, X_dev.shape:{1}, y_train.shape:{2}, y_dev.shape:{3}'.format(X_train.shape, X_dev.shape, y_train.shape, y_dev.shape))

    train(X_train, X_dev, y_train, y_dev)

if __name__ == "__main__":
    main()