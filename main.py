import os
import model
from settings import Settings
import argparse
import datetime
import downloader

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=2000)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=20)
parser.add_argument('--height', dest='height', type=int, default=32)
parser.add_argument('--width', dest='width', type=int, default=32)
parser.add_argument('--lr', dest='lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--model', dest='model_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
parser.add_argument('--dataset', dest='dataset_name', default='isf', choices=['gtsrb', 'isf'])
parser.add_argument('--lock', dest='training_lock', default='none', choices=['none', 'cnn', 'dnn', 'cnn-dnn'])

args = parser.parse_args()
train_path, test_path = downloader.get_necessary_data(args.dataset_name, '.' + os.sep + 'data')

# New settings
settings = Settings({
    'conv_filters': [32, 32, 64, 64, 128, 128],
    'conv_kernels': [2, 2, 2, 2, 2, 2],
    'pooling_after_conv': [False, True, False, True, False, True],
    'fc_hidden': [128, 128],
    'height': args.height,
    'width': args.width,
    'batch_size': args.batch_size,
    'learning_rate': args.lr,
    'training_lock': args.training_lock,
    'model_name': args.model_name,
    'train_data_dir': train_path,
    'test_data_dir': test_path,
})

# Load settings
# settings = Settings(None, restore_from_path='./models/deep-cnn/settings.txt')

# Continue training from loaded settings file
model.train(settings, n_epochs=args.epoch, restore_type='auto')

# Continue training from certain epoch out of the models save directory
# model.train(settings, n_epochs=args.epoch, restore_type='by_name', restore_data='epoch192')

# Load model from path and start training
# model.train(settings, n_epochs=args.epoch, restore_type='path', restore_data='./models/minimalistic-cnn/saves/epoch100.ckpt')

# Load only certain parts of model given by path and start training. Loaded parts will be locked for training (--lock)
# model.train(settings, n_epochs=args.epoch, restore_type='transfer', restore_data='./models/deep-cnn/saves/epoch119.ckpt')
