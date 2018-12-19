import argparse
import datetime
import sys
import warnings
import os
import model
import util
import tf_util
from data import DataManager, OrderedBatchProvider
from settings import Settings
import cv2
import numpy as np
import tensorflow as tf
import time

warnings.filterwarnings("ignore")


def sliding_window(settings, image, window_width, window_height, stride):
    x_iter = 0
    y_iter = 0
    windows = []
    while x_iter * stride + window_width < np.shape(image)[1]:
        while y_iter * stride + window_height < np.shape(image)[0]:
            window = image[y_iter*stride:y_iter*stride+window_height, x_iter*stride:x_iter*stride+window_width]
            if settings.get_setting_by_name('channels') == 1:
                window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY, )
            window = cv2.resize(window, (settings.get_setting_by_name('height'), settings.get_setting_by_name('width')))
            if len(np.shape(window)) < 3:
                window = np.expand_dims(window, 3)
            windows.append(window)
            # cv2.imshow('window', window)
            # cv2.waitKey(0)
            y_iter += 1
        y_iter = 0
        x_iter += 1

    return windows


parser = argparse.ArgumentParser()
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)
parser.add_argument('--test_data', dest='path_to_test_data', type=str, default=None)

args = parser.parse_args()

if args.path_to_settings is None:
    print('ERROR: need settings to load model from!')
    sys.exit(0)
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

data = util.read_any_data(args.path_to_test_data)
test_images = [data[200], data[400], data[600], data[800], data[1000]]

rectslist = []
for image in test_images:
    rectslist.append(util.get_rects_sliding_window(settings, image, 30, 30, 15))
    # cv2.imshow('original', image)
    # cv2.waitKey(0)
    # windows = sliding_window(settings, image, 60, 60, 10)
    # batch_provider = OrderedBatchProvider(windows, [], settings.get_setting_by_name('batch_size'))
    # tf_util.execute_frozen_model(settings, batch_provider)

print(np.shape(rectslist))
tf_util.execute_on_subimages(settings, test_images, rectslist)
