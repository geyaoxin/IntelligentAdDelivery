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

def timeConversion(data):
    #TODO: Convert the data in '时间' column to an integer from 0 to 24.
    '''
    For example:
        2020-11-02 17:00:04 --> 17
    '''
    handled_data=data.copy()
    ls=[str(i) for i in handled_data["时间"]]
    for i in range(len(ls)):
       handled_data["时间"][i]=ls[i].split(" ")[1].split(":")[0]

    return handled_data

def load_data():
    # load all data.
    xlsx_file = pd.read_excel('./data/NovData.xlsx')
    head_data = xlsx_file.head()
    print("These are the head 5 rows of all data: \n{0}".format(head_data))
    
    # Data cleaning: drop out unuseful data.
    cleaned_data = xlsx_file.drop(index=xlsx_file.点击量[xlsx_file.点击量==0].index)

    # Feature engineering: select those columns we want.
    cols_X = cleaned_data[['广告出价（分）', '订单单价，单位为“分”']]    #independent variable
    exposure = cleaned_data[['曝光量']]                                #dependent variable
    click_rate = cleaned_data[['点击率']]
    cost = cleaned_data[['花费，单位为“分”']]
    browse_rate = cleaned_data[['商品页浏览率']]
    order_rate = cleaned_data[['下单率']]
    roi = cleaned_data[['订单ROI']]

    #TODO: pick time data into the independent variable data.
    #cols_X = timeConversion(cleaned_data[['广告出价（分）', '订单单价，单位为“分”', '时间']])

    return cols_X, exposure, click_rate, cost, browse_rate, order_rate, roi