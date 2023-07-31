import numpy as np

def label_data(y):
    y[np.logical_and(y > 0, y < 0.25)] = 0
    two = 