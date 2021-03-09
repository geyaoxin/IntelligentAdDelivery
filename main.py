#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv
import pickle
import datetime
import time

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import VotingClassifier

from utils import *

def main(data):
    train_test_split(data, y, test_size=0.1, random_state=42, shuffle=True)

if __name__ == "__main__":
    args = args_parser()
    print("loading data...")
    data = load_data()
    main(data)
    ############################