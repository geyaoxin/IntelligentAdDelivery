#-*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import argparse
import pandas as pd
import csv
import pickle
import datetime
import time

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument('--epoch', type=int, default=100, help="epochs")
    parser.add_argument('--algo', type=str, default='linearRegression', help="regression algorithm")

    args = parser.parse_args()
    return args

def load_data():
    # load all data.
    xlsx_file = pd.read_excel('./data/NovData.xlsx')
    head_data = xlsx_file.head()
    print("These are the head 5 rows of all data: \n{0}".format(head_data))
    
    # Data cleaning: drop out unuseful data.
    cleaned_data = xlsx_file.drop(index=xlsx_file.点击量[xlsx_file.点击量==0].index)

    # Feature engineering: select those columns we want.
    cols = ['广告出价（分）', '订单单价，单位为“分”', '订单ROI', '曝光量', '点击率', '花费，单位为“分”', '商品页浏览率', '下单率']
    final_data = cleaned_data[cols]
    return final_data