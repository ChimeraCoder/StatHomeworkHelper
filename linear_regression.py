from __future__ import division
import numpy as np


def calculate_hat_matrix(x_vector):
    '''Calculate the hat matrix, which is defined as X(X'X)^-1X' '''
    #Currently only works for single regression
    #Assume X is a COLUMN vector
    #Calculate (x'x)^-1
    inverted = np.dot(x.T, x)
    h = np.dot(     np.dot( x,      1/np.dot(x.T, x)[0,0]        ), x.T)
    return h


def get_J(num_observations):
    '''Calculate J, the appropriate matrix of ones'''
    n = num_observations
    J = np.ones(n*n)
    J.shape = (n,n)
    return J


def calculate_SSR(x, y):
    '''Calculate the squared sum of the regression'''
    #Assume x and y are both column vectors
    n = x.shape[0]
    inner = calculate_hat_matrix(x) - 1/n* get_J(n) 
    result = np.dot( np.dot(y.T, inner) , y)
    return result


def calculate_SSE(x, y):
    '''Calculate the squared sum of the error'''
    H = calculate_hat_matrix(x)
    I = np.eye(x.shape[0])
    inner = I - H
    result = np.dot( np.dot(y.T, inner)  , y)
    return result


def calculate_MSE(x, y):
    '''Calculate the mean squared error'''
    n = x.shape[0]
    result = calculate_SSE(x,y)/(n-2)
    return result[0][0]


def calculate_variance_covariance_matrix(x, y):
    '''Calculate the variance-covariance matrix of the estimated regression coefficients'''
    mse = calculate_MSE(x,y)
    matrix = 1 / np.dot( x.T, x)
    return mse*matrix


def calculate_variance_covariance_matrix_of_residuals(x, y):
    '''Calculate the variance-covariance matrix of the residuals'''
    H = calculate_hat_matrix(x)
    I = np.eye(x.shape[0])
    mse = calculate_MSE(x,y)
    return mse * (I-H)


