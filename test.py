import argparse
import sys
import warnings
import os
import util
import tf_util
from data import DataManager
from settings import Settings
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='dataset_name', default='sliding_window', choices=['sliding_window', 'test'])
parser.add_argument('--method', dest='method', default='sliding_window', choices=['sliding_window', 'regular'])
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)

args = parser.parse_args()

if args.path_to_settings is None:
    print('ERROR: need settings to load model from!')
    sys.exit(0)
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

path_to_test_data = util.get_necessary_test_data(args.dataset_name, '.' + os.sep + 'data')
if args.method == 'sliding_window':
    data = util.read_any_data(path_to_test_data)[:-1]
    data = [data[200], data[400], data[600], data[800], data[1000]]

    rectangle_list = []
    for image in data:
        rectangle_list.append(util.get_rects_sliding_window(image, 60, 60, 60))#

    print('shape of structure holding rects: ', np.shape(rectangle_list))
    tf_util.execute_on_subimages(settings, data, rectangle_list)
elif args.method == 'regular':
    print('regular')
    settings.update({'train_data_dir': '', 'test_data_dir': '.' + os.sep + 'data' + os.sep + args.dataset_name}, write_to_file=False)
    test_data_provider = DataManager(settings).test_provider
    tf_util.execute_frozen_model(settings, DataManager(settings).test_provider)
else:
    print('ERROR: invalid method')
