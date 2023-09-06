

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error
import os
import sys
import math


parent_dir = os.path.dirname(os.path.realpath('../../analysis'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Import the module from the parent directory
import analysis.model_assessment
import analysis.train_data_analysis
from analysis.model_assessment import *
from scipy.stats import linregress


def plot_scatter_permutations(fnames, mode):

    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    
    row = 0
    col = 0
    
    for i in range(len(fnames)):
    
        if col == 2:
            col = 0
            row+=1
    
        path = fnames[i]
        data = np.load(path)
        y_pred, y_test, y_train = data['arr_0'], data['arr_1'], data['arr_2']
    
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        
        crosstrack_pixel_length = 1242
        num_test_scenes = math.floor(y_test.shape[0] / crosstrack_pixel_length)
        y_test = y_test.reshape((num_test_scenes, crosstrack_pixel_length))
        y_pred = y_pred.reshape((num_test_scenes, crosstrack_pixel_length))

        if mode == 'percentile':
        
            test_2 = np.percentile(y_test, 0.02, axis=1)
            pred_2 = np.percentile(y_pred, 0.02, axis=1)
            test_98 = np.percentile(y_test, 0.98, axis=1)
            pred_98 = np.percentile(y_pred, 0.98, axis=1)
            
            test_ranges = np.subtract(test_98, test_2)
            pred_ranges = np.subtract(pred_98, pred_2)
            
            title = 'WVR Percentile Comparison for ' + model_name
    
            mse = mean_squared_error(test_ranges, pred_ranges)
            slope, intercept, r_value, p_value, std_err = linregress(test_ranges, pred_ranges)

            axs[row, col].scatter(test_ranges, pred_ranges, alpha = 0.5)
        
        else:
    
            test_ranges = np.max(y_test, axis = 1) - np.min(y_test, axis = 1)
            pred_ranges = np.max(y_pred, axis = 1) - np.min(y_pred, axis = 1)
    
            title = 'WVR Absolute Comparison for ' + model_name
    
            mse = mean_squared_error(test_ranges, pred_ranges)
            slope, intercept, r_value, p_value, std_err = linregress(test_ranges, pred_ranges)

            axs[row, col].scatter(test_ranges, pred_ranges, alpha = 0.5)
            axs[row, col].set_xticks(range(0, 7))
            axs[row, col].set_yticks(range(0, 7))
            
        axs[row, col].text(0.7, 0.15,  'y = %.2fx + %.2f' % (slope, intercept), transform=axs[row, col].transAxes)
        axs[row, col].text(0.7, 0.1,  'MSE: %.2f' % (mse), transform=axs[row, col].transAxes)
        axs[row, col].set_title(title);
        axs[row, col].set_xlabel('Actual WV Ranges (g/cm^2)');
        axs[row, col].set_ylabel('Predicted WV Ranges (g/cm^2)');
    
        col+=1


def plot_hist_train_test(fnames):

    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    
    row = 0
    col = 0
    
    for i in range(len(fnames)):
    
        if col == 2:
            col = 0
            row+=1
    
        path = fnames[i]
        data = np.load(path)
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        y_pred, y_test, y_train = data['arr_0'], data['arr_1'], data['arr_2']

        mse = mean_squared_error(y_test, y_pred)
        #print('Mean Squared Error: ', mse)
        
        axs[row, col].hist(y_test, alpha = 0.6, label = 'Test')
        axs[row, col].hist(y_pred, alpha = 0.6, label = 'Predicted')
        axs[row, col].set_xlabel('Atmospheric WV (g/cm^2)')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_title('Train and Test Pixels for ' + model_name)
        axs[row, col].text(0.8, 0.5,  'MSE: %.2f' % (mse), transform=axs[row, col].transAxes)
        axs[row, col].legend()

        col+=1


def plot_scatter_test_pred(fnames):

    fig, axs = plt.subplots(2, 2, figsize = (10, 8))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    
    row = 0
    col = 0
    
    for i in range(len(fnames)):
    
        if col == 2:
            col = 0
            row+=1
    
        path = fnames[i]
        data = np.load(path)
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        y_pred, y_test, y_train = data['arr_0'], data['arr_1'], data['arr_2']
        print(y_test.shape)
        print(y_pred.shape)

        mse = mean_squared_error(y_test, y_pred)
        #print('Mean Squared Error: ', mse)
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        #print('R-Squared of Fit: ', r_value)
        
        axs[row, col].scatter(y_test, y_pred, alpha = 0.1)
        axs[row, col].set_xticks(range(0, 7))
        axs[row, col].set_yticks(range(0, 7))
        axs[row, col].text(0.7, 0.15,  'y = %.2fx + %.2f' % (slope, intercept), transform=axs[row, col].transAxes)
        axs[row, col].text(0.7, 0.1,  'MSE: %.2f' % (mse), transform=axs[row, col].transAxes)
        axs[row, col].set_title('Per-Pixel Performance of ' + model_name);
        axs[row, col].set_xlabel('Actual WV (g/cm^2)');
        axs[row, col].set_ylabel('Predicted WV (g/cm^2)');

        col+=1


def plot_scatter_test_pred_all_models(fnames):

    fig, axs = plt.subplots(1, 3, figsize = (12, 4))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    col = 0
    
    for i in range(len(fnames)):
    
    
        path = fnames[i]
        data = np.load(path)
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        y_pred, y_test, y_train = data['arr_0'], data['arr_1'], data['arr_2']
        print(y_test.shape)
        print(y_pred.shape)

        mse = mean_squared_error(y_test, y_pred)
        #print('Mean Squared Error: ', mse)
        slope, intercept, r_value, p_value, std_err = linregress(y_test, y_pred)
        #print('R-Squared of Fit: ', r_value)
        
        axs[col].scatter(y_test, y_pred, alpha = 0.1)
        axs[col].set_xticks(range(0, 7))
        axs[col].set_yticks(range(0, 7))
        axs[col].text(0.7, 0.15,  'y = %.2fx + %.2f' % (slope, intercept), transform=axs[col].transAxes)
        axs[col].text(0.7, 0.1,  'MSE: %.2f' % (mse), transform=axs[col].transAxes)
        axs[col].set_title('Per-Pixel Performance of ' + model_name);
        axs[col].set_xlabel('Actual WV (g/cm^2)');
        axs[col].set_ylabel('Predicted WV (g/cm^2)');

        col+=1


def plot_range_scatter_all_models(fnames, mode):

    fig, axs = plt.subplots(1, 3, figsize = (12, 4))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    
    col = 0
    
    for i in range(len(fnames)):
    
        path = fnames[i]
        data = np.load(path)
        y_pred, y_test, y_train = data['arr_0'], data['arr_1'], data['arr_2']
    
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        
        crosstrack_pixel_length = 1242
        num_test_scenes = math.floor(y_test.shape[0] / crosstrack_pixel_length)
        y_test = y_test.reshape((num_test_scenes, crosstrack_pixel_length))
        y_pred = y_pred.reshape((num_test_scenes, crosstrack_pixel_length))

        if mode == 'percentile':
        
            test_2 = np.percentile(y_test, 0.02, axis=1)
            pred_2 = np.percentile(y_pred, 0.02, axis=1)
            test_98 = np.percentile(y_test, 0.98, axis=1)
            pred_98 = np.percentile(y_pred, 0.98, axis=1)
            
            test_ranges = np.subtract(test_98, test_2)
            pred_ranges = np.subtract(pred_98, pred_2)
            
            title = 'WVR Percentile Comparison for ' + model_name
    
            mse = mean_squared_error(test_ranges, pred_ranges)
            slope, intercept, r_value, p_value, std_err = linregress(test_ranges, pred_ranges)

            axs[col].scatter(test_ranges, pred_ranges, alpha = 0.5)
        
        else:
    
            test_ranges = np.max(y_test, axis = 1) - np.min(y_test, axis = 1)
            pred_ranges = np.max(y_pred, axis = 1) - np.min(y_pred, axis = 1)
    
            title = 'WVR Absolute Comparison for ' + model_name
    
            mse = mean_squared_error(test_ranges, pred_ranges)
            slope, intercept, r_value, p_value, std_err = linregress(test_ranges, pred_ranges)

            axs[col].scatter(test_ranges, pred_ranges, alpha = 0.5)
            axs[col].set_xticks(range(0, 7))
            axs[col].set_yticks(range(0, 7))
            
        axs[col].text(0.7, 0.15,  'y = %.2fx + %.2f' % (slope, intercept), transform=axs[col].transAxes)
        axs[col].text(0.7, 0.1,  'MSE: %.2f' % (mse), transform=axs[col].transAxes)
        axs[col].set_title(title);
        axs[col].set_xlabel('Actual WV Ranges (g/cm^2)');
        axs[col].set_ylabel('Predicted WV Ranges (g/cm^2)');
    
        col+=1


def plot_range_scatter_all_models_pixel_cloud_mask(fnames):

    fig, axs = plt.subplots(1, 3, figsize = (12, 4))
    fig.tight_layout(pad=0.4, w_pad=2.5, h_pad=4.0)
    
    col = 0
    
    for i in range(len(fnames)):
    
        path = fnames[i]
        data = np.load(path)
        y_pred, y_test, y_test_placehold, y_test_cloud_mask, y_train = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4']
    
        model_name = path[path.rfind('/')+1: path.rfind('.')].upper()
        y_pred_resh = np.zeros(y_test_placehold.shape)
        y_pred_resh = y_pred_resh.flatten()
        y_pred_resh[y_test_cloud_mask.flatten() == 0] = y_pred
        y_pred_resh = y_pred_resh.reshape(y_test_placehold.shape)

        test_ranges = np.nanmax(y_test_placehold, axis = 1) - np.nanmin(y_test_placehold, axis = 1)
        pred_ranges = np.nanmax(y_pred_resh, axis = 1) - np.nanmin(y_pred_resh, axis = 1)

        title = 'WVR Absolute Comparison for ' + model_name

        mse = mean_squared_error(test_ranges, pred_ranges)
        slope, intercept, r_value, p_value, std_err = linregress(test_ranges, pred_ranges)

        axs[col].scatter(test_ranges, pred_ranges, alpha = 0.5)
        axs[col].set_xticks(range(0, 7))
        axs[col].set_yticks(range(0, 7))
            
        axs[col].text(0.7, 0.15,  'y = %.2fx + %.2f' % (slope, intercept), transform=axs[col].transAxes)
        axs[col].text(0.7, 0.1,  'MSE: %.2f' % (mse), transform=axs[col].transAxes)
        axs[col].set_title(title);
        axs[col].set_xlabel('Actual WV Ranges (g/cm^2)');
        axs[col].set_ylabel('Predicted WV Ranges (g/cm^2)');
    
        col+=1