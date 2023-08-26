import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns

def random_select(y_test, y_pred, indices):
    test = np.array(y_test)[indices]
    pred = np.array(y_pred)[indices]
    return test, pred


def plot_bar_comparison(test, pred, title):
    plt.bar(x = np.arange(test.shape[0]), height = test, alpha = 0.5, label = 'Actual')
    plt.bar(x = np.arange(test.shape[0]), height = pred, alpha = 0.5, label = 'Predicted')
    plt.title(title)
    plt.legend()

def plot_residuals(test, pred, title):
    residuals = abs(test - pred)
    plt.scatter(np.arange(test.shape[0]), residuals)
    plt.title(title)

def plot_hist(set_1, set_2, title, description_1, description_2):
    plt.hist(set_1, alpha = 0.7, label = description_1)
    plt.hist(set_2, alpha = 0.7, label = description_2)
    plt.legend()
    plt.xlabel('Atmospheric WV (g/cm^2)')
    plt.ylabel('Frequency')
    plt.title(title)

def plot_scatter(y_test, y_pred, title):
    plt.scatter(y_test, y_pred, alpha = 0.5)
    plt.xlabel('g/cm^2')
    plt.ylabel('g/cm^2')
    plt.annotate("r-squared = {:.3f}".format(r2_score(y_test, y_pred)), (1, 5))
    plt.title(title)

def plot_hist_and_scatter(y_test, y_pred, title_hist, title_scatter):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4), constrained_layout = True)
    axs[0].hist(y_test, alpha = 0.7, label = 'Actual ')
    axs[0].hist(y_pred, alpha = 0.7, label = 'Predicted')
    axs[0].legend()
    axs[0].set_xlabel('Atmospheric WV (g/cm^2)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(title_hist)

    axs[1].scatter(y_test, y_pred, alpha = 0.5)
    axs[1].annotate("r-squared = {:.3f}".format(r2_score(y_test, y_pred)), (1, 5))
    axs[1].set_xlabel('g/cm^2')
    axs[1].set_ylabel('g/cm^2')
    axs[1].set_title(title_scatter)

def compare_train_test_pred(y_train, y_test, y_pred, model_name):
    plt.hist(y_train, alpha = 0.6, label = 'Train')
    plt.hist(y_test, alpha = 0.6, label = 'Test')
    plt.hist(y_pred, alpha = 0.6, label = 'Predicted')
    plt.xlabel('Atmospheric WV (g/cm^2)')
    plt.ylabel('Frequency')
    plt.title('Train, Test and Predicted Histograms for ' + model_name)
    plt.legend()

def compare_test_pred(y_test, y_pred, model_name):
    plt.hist(y_test, alpha = 0.6, label = 'Test')
    plt.hist(y_pred, alpha = 0.6, label = 'Predicted')
    plt.xlabel('Atmospheric WV (g/cm^2)')
    plt.ylabel('Frequency')
    plt.title('Train, Test and Predicted Histograms for ' + model_name)
    plt.legend()

def plot_bar_and_residuals(test, pred, title_bar, title_residuals):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4), constrained_layout = True)
    
    axs[0].bar(x = np.arange(test.shape[0]), height = test, alpha = 0.5, label = 'Actual')
    axs[0].bar(x = np.arange(test.shape[0]), height = pred, alpha = 0.5, label = 'Predicted')
    axs[0].set_title(title_bar)
        
    residuals = abs(test - pred)
    axs[1].scatter(np.arange(test.shape[0]), residuals)
    axs[1].set_title(title_residuals)

        


def plot_range_hist(y_test, y_pred, path, mode):

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

        title = 'WV Range Values (2nd to 98 percentiles) for ' + model_name

    else:
    
        test_ranges = np.max(y_test, axis = 1) - np.min(y_test, axis = 1)
        pred_ranges = np.max(y_pred, axis = 1) - np.min(y_pred, axis = 1)

        title = 'WV Range Values (Abs Max - Abs Min) for ' + model_name
        
    mse = mean_squared_error(test_ranges, pred_ranges)
    print('Mean Squared Error: ', mse)
    plt.hist(test_ranges, alpha = 0.7, label = 'Test');
    plt.hist(pred_ranges, alpha = 0.7, label = 'Predicted');
    plt.title(title);
    plt.xlabel('WV Range (g/cm^2)');
    plt.ylabel('Frequency');
    plt.legend();

def plot_range_scatter(y_test, y_pred, path, mode):
    
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

        title = 'WV Range Scatter (2nd to 98 percentiles) for ' + model_name

    else:
    
        test_ranges = np.max(y_test, axis = 1) - np.min(y_test, axis = 1)
        pred_ranges = np.max(y_pred, axis = 1) - np.min(y_pred, axis = 1)

        title = 'WV Range Scatter (Abs Max - Abs Min) for ' + model_name

    mse = mean_squared_error(test_ranges, pred_ranges)
    r2 = r2_score(test_ranges, pred_ranges)
    print('Mean Squared Error: ', mse)
    print('R-Squared of Fit: ', r2)

    #m, b = np.polyfit(test_ranges, pred_ranges, 1)
    #plt.plot(test_ranges, m*test_ranges+b)
    plt.scatter(test_ranges, pred_ranges, alpha = 0.5)
    plt.title(title);
    plt.xlabel('Actual WV Ranges (g/cm^2)');
    plt.ylabel('Predicted WV Ranges (g/cm^2)');
    plt.legend();

def get_percentiles(y_test, y_pred):

    crosstrack_pixel_length = 1242
    num_test_scenes = math.floor(y_test.shape[0] / crosstrack_pixel_length)
    y_test = y_test.reshape((num_test_scenes, crosstrack_pixel_length))
    y_pred = y_pred.reshape((num_test_scenes, crosstrack_pixel_length))
    
    test_2 = np.percentile(y_test, 0.02, axis=1)
    pred_2 = np.percentile(y_pred, 0.02, axis=1)
    test_98 = np.percentile(y_test, 0.98, axis=1)
    pred_98 = np.percentile(y_pred, 0.98, axis=1)

    return test_2, pred_2, test_98, pred_98



