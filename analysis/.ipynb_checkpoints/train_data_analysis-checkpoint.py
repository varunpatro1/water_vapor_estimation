import pandas as pdf
import numpy as np
import matplotlib.pyplot as plt

def plot_overall_hist(wv_by_scene):
    plt.hist(wv_by_scene.flatten())
    plt.xlabel('WV (g/cm^2)')
    plt.ylabel('Frequency')
    plt.title('WV Histogram EMIT Data')

def colorbar_all_crosstracks(wv_by_scene):
    plt.imshow(wv_by_scene, vmin = -1, vmax = 8)
    plt.colorbar()
    yticks = range(0, 750, 75)
    plt.yticks(yticks)
    plt.xticks(range(0, 1300, 100), rotation = 'vertical')
    plt.xlabel('Crosstrack Pixel Number')
    plt.ylabel('Scene Number')
    plt.title('WV for Cloud Filtered Individual Scene Crosstracks')

def plot_all_crostracks(wv_by_scene):
    for i in range(wv_by_scene.shape[0]):
        plt.plot(wv_by_scene[i])
        plt.yticks(range(-1, 8))
        plt.xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
        plt.ylabel('Water Vapor (g/cm^2)')
        plt.xlabel('Crosstrack Pixel Number')
        plt.title('Atmospheric WV for Cloud Filtered Individual Scene Downtracks')

