import argparse
import datetime
import sys
import warnings
import os
import model
import util
import tf_util
from data import DataManager
from settings import Settings
import cv2

warnings.filterwarnings("ignore")

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

for image in data:
    cv2.imshow('original', image)
    cv2.waitKey(0)


# def test_frozen_model(settings, data_manager):
#     for i in range(len(test_batch_x)):
#         windows = util.sliding_window(image=test_batch_x[i], width=settings.get_setting_by_name('width'), height=settings.get_setting_by_name('height'), stride=10)
#
#
# tf_util.execute_frozen_model(settings, data_manager)