import matplotlib.pyplot as plt
import numpy as np

def random_select(y_test, y_pred, num_to_select):
    indices = np.random.randint(0, y_test.shape[0], num_to_select)
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

def plot_hist(y_test, y_pred, title):
    plt.hist(y_test, alpha = 0.7, label = 'Actual ')
    plt.hist(y_pred, alpha = 0.7, label = 'Predicted')
    plt.legend()
    plt.xlabel('Atmospheric WV (g/cm^2)')
    plt.ylabel('Frequency')
    plt.title(title)



def plot_bar_and_residuals(test, pred, title_bar, title_residuals):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (8,4), constrained_layout = True)
    
    axs[0].bar(x = np.arange(test.shape[0]), height = test, alpha = 0.5, label = 'Actual')
    axs[0].bar(x = np.arange(test.shape[0]), height = pred, alpha = 0.5, label = 'Predicted')
    axs[0].set_title(title_bar)
        
    residuals = abs(test - pred)
    axs[1].scatter(np.arange(test.shape[0]), residuals)
    axs[1].set_title(title_residuals)



