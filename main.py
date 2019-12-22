import argparse
import datetime
import sys
import warnings
import model
import util
import tf_util
from data import DataManager, OrderedBatchProvider
from settings import Settings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# Number of epochs of training. (One epoch uses all training material)
parser.add_argument('--epoch', dest='epoch', type=int, default=20)
# Batch size for single training inference
parser.add_argument('--batch_size', dest='batch_size', type=int, default=160)
# Image height that neural net expects. (Images of differing sizes will be scaled)
parser.add_argument('--height', dest='height', type=int, default=32)
# Image width that neural net expects. (Images of differing sizes will be scaled)
parser.add_argument('--width', dest='width', type=int, default=32)
# Number of channels that the neural net expects. (If channels=1, conversion to grayscale is applied if necessary)
parser.add_argument('--channels', dest='channels', type=int, default=3)
# Initial learning rate
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.0001)
# Learning rate decay (Factor with which learning rate decreases every epoch)
parser.add_argument('--lr_decay', dest='learning_rate_decay', type=float, default=0.99)
# Dropout, only applied to dense layers (Dropout=percentage of neurons randomly omitted in training)
parser.add_argument('--dropout', dest='dropout_probability', type=float, default=1.0)
# Whether to use batch norm, only applied to convolutional layers
parser.add_argument('--batch_norm', dest='batch_norm', type=bool, default=False, choices=[True, False])
# The name of the trained model (used for saving the model at desired path)
parser.add_argument('--model', dest='model_name', default=datetime.datetime.now().strftime("%I_%M%p_on_%B_%d,_%Y"))
# The name of the dataset used for training. If available, the dataset will be downloaded automatically.
# isf -> limited dataset with 4 signs, badly cropped.
# isf-new -> also limited, better crop, but strange lighting (red shimmer)
# isf-complete -> complete dataset, all signs, various lighting, precisely cropped
# gtsrb -> real street signs, (Not all classes from CC present, even contains new classes)
# mnist -> standard hand written digits dataset
# cifar10 -> standard 32x32 image classification dataset (may be used for pre-training a model)
parser.add_argument('--dataset', dest='dataset_name', default='isf-complete', choices=['isf', 'isf-new', 'isf-complete', 'gtsrb', 'mnist', 'cifar', 'cifar100'])
# Whether to augment the available training data at the start of the training
parser.add_argument('--augment', dest='augment_dataset', type=bool, default=False, choices=[True, False])
# For each real image, how many augmented images should be generated? (Program keeps track of original images via
# index file. If scalar is decreased after initial augmentation, augmented images will be deleted)
parser.add_argument('--augment_scalar', dest='augment_scalar', type=float, default=2.0)
# Path to settings file. (Settings will be restored from the given file)
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)
# How to restore a previously trained model
# auto -> If a settings file was specified (--settings), automatically resume the corresponding training
# by_name -> find a model in the ./models folder by name specified via (--restore_argument)
# path -> restore a model through its .ckpt files given by path path/to/model/files.ckpt via (--restore_arguent)
# transfer -> Same as path, but only certain parts of the model (given by --lock) will be restored
# NOTICE: If not using auto, the current settings have to match with the settings of the model you want to restore!
parser.add_argument('--restore', dest='restore_type', default='auto', choices=['auto', 'by_name', 'path', 'transfer'])
# Multi purpose argument used for restoring a model, for usage see explanation of (--restore)
parser.add_argument('--restore_argument', dest='restore_argument', default='')
# If 'transfer' is used with (--restore), this argument decides which parts of the model will be restored and locked
parser.add_argument('--lock', dest='training_lock', type=str, default='none', choices=['none', 'cnn', 'dnn', 'cnn-dnn'])
# Seed for stochastic processes for better reproducibility
parser.add_argument('--seed', dest='seed', type=int, default=0)
# Name of final node in inference graph
parser.add_argument('--output_node_name', dest='output_node_name', type=str, default='output_soft')
# Name of input node in inference graph
parser.add_argument('--input_node_name', dest='input_node_name', type=str, default='input_placeholder')
# Skip training in execution
parser.add_argument('--no-train', dest='train', action='store_false', default=True)
# Freeze model after training
parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
# Export model (creates zip with settings and frozen model)
parser.add_argument('--export', dest='export', action='store_true', default=False)
# Execute trained model (shows images and their classifications)
parser.add_argument('--execute', dest='execute', action='store_true', default=False)
# Whether to use synthetic data for training (For each batch new images will be generated by randomly placing
# a foreground image (--foreground), on a background image/video (--background) and randomly cropping. A maximum number
# of synthetic images per batch can be specified via (--max_synthetic_per_epoch))
parser.add_argument('--synthetic_data', dest='use_synthetic_training_data', action='store_true', default=False)
# Path to folder with background images/videos
parser.add_argument('--background', dest='path_to_background_data', type=str, default=None)
# Path to folder with foreground images
parser.add_argument('--foreground', dest='path_to_foreground_data', type=str, default=None)
# Maximum number of synthetic images per batch. Remaining batch size will be used by real training data.
parser.add_argument('--max_synthetic_per_epoch', dest='maximum_synthetic_batches_per_epoch', type=int, default=50)


args = parser.parse_args()
train_path, test_path = util.get_necessary_dataset(args.dataset_name, Settings.data_path_from_root)

if args.augment_dataset:
    util.augment_data(scalar=args.augment_scalar, path_to_data=train_path, path_to_index=util.get_index(train_path), balance=True)

if args.path_to_settings is None:
    settings = Settings({
        'conv_filters': [32, 32, 64, 64, 128],
        'conv_kernels': [3, 3, 2, 2, 2],
        'pooling_after_conv': [False, True, False, True, True],
        'fc_hidden': [1024],
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
        'input_node_name': args.input_node_name,
        'output_node_name': args.output_node_name,
        'train_data_dir': train_path,
        'test_data_dir': test_path,
        'use_synthetic_training_data': args.use_synthetic_training_data,
        'path_to_background_data': args.path_to_background_data,
        'path_to_foreground_data': args.path_to_foreground_data,
        'maximum_synthetic_batches_per_epoch': args.maximum_synthetic_batches_per_epoch
    })
else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

settings.assess(args)
data_manager = DataManager(settings)

if args.train:
    try:
        model.train(settings, data_manager, n_epochs=args.epoch, restore_type=args.restore_type)
    except KeyboardInterrupt:
        print('Stop training..')

if args.freeze:
    tf_util.freeze_graph(settings)

if args.export:
    util.export_model_to_production(settings)

if args.execute:
    tf_util.execute_frozen_model(settings, data_manager.test_provider)
