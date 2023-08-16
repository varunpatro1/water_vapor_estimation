import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

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



def plot_bar_and_residuals(test, pred, title_bar, title_residuals):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4), constrained_layout = True)
    
    axs[0].bar(x = np.arange(test.shape[0]), height = test, alpha = 0.5, label = 'Actual')
    axs[0].bar(x = np.arange(test.shape[0]), height = pred, alpha = 0.5, label = 'Predicted')
    axs[0].set_title(title_bar)
        
    residuals = abs(test - pred)
    axs[1].scatter(np.arange(test.shape[0]), residuals)
    axs[1].set_title(title_residuals)



