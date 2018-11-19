import argparse
import datetime
import warnings
import os
import model
import util
import tf_util
from settings import Settings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', dest='epoch', type=int, default=100)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
parser.add_argument('--height', dest='height', type=int, default=32)
parser.add_argument('--width', dest='width', type=int, default=32)
parser.add_argument('--channels', dest='channels', type=int, default=3)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001)
parser.add_argument('--lr_decay', dest='learning_rate_decay', type=float, default=0.99)
parser.add_argument('--dropout', dest='dropout_probability', type=float, default=1.0)
parser.add_argument('--batch_norm', dest='batch_norm', type=bool, default=False, choices=[True, False])
parser.add_argument('--model', dest='model_name', default=datetime.datetime.now().strftime("%I_%M%p_on_%B_%d,_%Y"))
parser.add_argument('--dataset', dest='dataset_name', default='isf', choices=['gtsrb', 'isf', 'mnist'])
parser.add_argument('--augment', dest='augment_dataset', type=bool, default=False, choices=[True, False])
parser.add_argument('--augment_scalar', dest='augment_scalar', type=float, default=2.0)
parser.add_argument('--restore', dest='restore_type', default='auto', choices=['auto', 'by_name', 'path', 'transfer'])
parser.add_argument('--restore_argument', dest='restore_argument', default='')
parser.add_argument('--lock', dest='training_lock', type=str, default='none', choices=['none', 'cnn', 'dnn', 'cnn-dnn'])
parser.add_argument('--seed', dest='seed', type=int, default=0)
parser.add_argument('--output_node_name', dest='output_node_name', type=str, default='output_soft')
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)

args = parser.parse_args()
train_path, test_path = util.get_necessary_data(args.dataset_name, '.' + os.sep + 'data')

if args.augment_dataset:
    util.augment_data(scalar=args.augment_scalar, path_to_data=train_path, path_to_index=util.get_index(train_path), balance=False)

if args.path_to_settings is None:
    settings = Settings({
        'conv_filters': [16, 16, 16, 16, 32, 32, 32],
        'conv_kernels': [5, 5, 5, 5, 2, 2, 2],
        'pooling_after_conv': [False, False, False, False, True, True, True],
        'fc_hidden': [128, 128],
        'height': args.height,
        'width': args.width,
        'channels': args.channels,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'learning_rate_decay': args.learning_rate_decay,
        'dropout': args.dropout_probability,
        'batch_norm': args.batch_norm,
        'training_lock': args.training_lock,
        'model_name': args.model_name,
        'output_node_name': args.output_node_name,
        'train_data_dir': train_path,
        'test_data_dir': test_path
    })
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

settings.assess(args)

try:
    model.train(settings, n_epochs=args.epoch, restore_type=args.restore_type)
except KeyboardInterrupt:
    print('exiting..')

tf_util.freeze_graph(settings)
