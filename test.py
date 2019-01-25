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
parser.add_argument('--data', dest='path_to_data', default='data'+os.sep+'isf'+os.sep+'test')
parser.add_argument('--method', dest='method', default='regular-no-labels', choices=['sliding_window', 'regular', 'regular-no-labels'])
parser.add_argument('--auto', dest='auto', action='store_true', default=False)
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)

args = parser.parse_args()

if args.auto:
    settings = util.import_latest_model()
elif args.path_to_settings is None:
    print('ERROR: need settings to load model from!')
    sys.exit(0)
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

if args.method == 'sliding_window':
    data = util.read_any_data(args.path_to_data)

    rectangle_list = []
    for image in data:
        rectangle_list.append(util.get_rects_sliding_window(image, 60, 60, 60))

    print('shape of structure holding rects: ', np.shape(rectangle_list))
    tf_util.execute_on_subimages(settings, data, rectangle_list)
elif args.method == 'regular':
    print('regular')
    settings.update({'train_data_dir': '', 'test_data_dir': args.path_to_data}, write_to_file=False)
    test_data_provider = DataManager(settings).test_provider
    tf_util.execute_frozen_model(settings, DataManager(settings).test_provider)
elif args.method == 'regular-no-labels':
    print('regular without labels')
    path_to_data = args.path_to_data
    data = util.read_any_data(path_to_data, settings=settings)
    test_data_provider = OrderedBatchProvider(data, [], settings.get_setting_by_name('batch_size'), [])
    # test_data_provider = DataManager(settings).test_provider
    tf_util.execute_frozen_model(settings, test_data_provider)
