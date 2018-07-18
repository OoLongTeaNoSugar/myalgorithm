#encoding: utf-8

# 逻辑回归
from numpy import *
import matplotlib as plt
import time
import pandas as pd
from sklearn import linear_model


def sigmod(theta,x):
    return 1.0/(1+exp(-theta.T*x))

# load data
train_data = pd.read_csv()

def costfunction(theta,x,y):
    J = 0
    grad = zeros(size(theta))
    m = len(x)

    J = -1.0 * sum(y*log(sigmod(theta,x))+(1.0 - y)*log(1-sigmod(theta,x))) / m


    return J, grad