import os
import sys

import model
from settings import Settings
import argparse
import datetime
import util
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=2000)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=500)
parser.add_argument('--height', dest='height', type=int, default=32)
parser.add_argument('--width', dest='width', type=int, default=32)
parser.add_argument('--channels', dest='channels', type=int, default=1)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001)
parser.add_argument('--lr_decay', dest='learning_rate_decay', type=float, default=0.99)
parser.add_argument('--dropout', dest='dropout_probability', type=float, default=0.8)
parser.add_argument('--model', dest='model_name', default=datetime.datetime.now().strftime("%I_%M%p_on_%B_%d,_%Y"))
parser.add_argument('--dataset', dest='dataset_name', default='isf', choices=['gtsrb', 'isf', 'mnist'])
parser.add_argument('--augment', dest='augment_dataset', type=bool, default=False, choices=[True, False])
parser.add_argument('--lock', dest='training_lock', default='none', choices=['none', 'cnn', 'dnn', 'cnn-dnn'])

args = parser.parse_args()
train_path, test_path = util.get_necessary_data(args.dataset_name, '.' + os.sep + 'data')

if args.augment_dataset:
    util.augment_data(scalar=4, data_dir=train_path, balance=True)

# New settings
settings = Settings({
    'conv_filters': [16, 16, 32, 32, 64, 64, 128, 128],
    'conv_kernels': [2, 2, 3, 2, 2, 2, 2, 2],
    'pooling_after_conv': [False, True, False, True, False, True, False, True],
    'fc_hidden': [128],
    'height': args.height,
    'width': args.width,
    'channels': args.channels,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'learning_rate_decay': args.learning_rate_decay,
    'dropout': args.dropout_probability,
    'training_lock': args.training_lock,
    'model_name': args.model_name,
    'train_data_dir': train_path,
    'test_data_dir': test_path,
})

# Load settings
# settings = Settings(None, restore_from_path='./models/mnist-test/settings.txt')

try:
    # Continue train from loaded settings file
    model.train(settings, n_epochs=args.epoch, restore_type='auto')

    # Continue train from certain epoch out of the models save directory
    # model.train(settings, n_epochs=args.epoch, restore_type='by_name', restore_data='epoch192')

    # Load model from path and start train
    # model.train(settings, n_epochs=args.epoch, restore_type='path', restore_data='./models/minimalistic-cnn/saves/epoch100.ckpt')

    # Load only certain parts of model given by path and start train. Loaded parts will be locked for train (--lock)
    # model.train(settings, n_epochs=args.epoch, restore_type='transfer', restore_data='./models/deep-cnn/saves/epoch119.ckpt')
except KeyboardInterrupt:
    print('exiting..')
