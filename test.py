import argparse
import sys
import warnings
import os
import util
import tf_util
from data import DataManager, OrderedBatchProvider
from settings import Settings
import numpy as np

# Beispielaufraufe:
# --settings ./models/isfbig/settings.txt --method regular --data test

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest='data_dir_name', default='sliding_window')
parser.add_argument('--method', dest='method', default='regular', choices=['sliding_window', 'regular', 'regular-no-labels'])
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)

args = parser.parse_args()

if args.path_to_settings is None:
    print('ERROR: need settings to load model from!')
    sys.exit(0)
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

if args.method == 'sliding_window':
    path_to_test_data = util.get_necessary_data(args.data_dir_name, '.' + os.sep + 'data')
    data = util.read_any_data(path_to_test_data)
    # data = [data[200], data[400], data[600], data[800], data[1000]]

    rectangle_list = []
    for image in data:
        rectangle_list.append(util.get_rects_sliding_window(image, 60, 60, 60))

    print('shape of structure holding rects: ', np.shape(rectangle_list))
    tf_util.execute_on_subimages(settings, data, rectangle_list)
elif args.method == 'regular':
    print('regular')
    settings.update({'train_data_dir': '', 'test_data_dir': '.' + os.sep + 'data' + os.sep + args.data_dir_name}, write_to_file=False)
    test_data_provider = DataManager(settings).test_provider
    tf_util.execute_frozen_model(settings, DataManager(settings).test_provider)
elif args.method == 'regular-no-labels':
    print('regular without labels')
    path_to_data = '.' + os.sep + 'data' + os.sep + args.data_dir_name
    data = util.read_any_data(path_to_data, settings=settings)
    test_data_provider = OrderedBatchProvider(data, [], settings.get_setting_by_name('batch_size'), [])
    # test_data_provider = DataManager(settings).test_provider
    tf_util.execute_frozen_model(settings, test_data_provider)
