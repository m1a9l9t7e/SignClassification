import logging
import argparse
import datetime
import warnings
import os
import cv2
import util
import model_util
from data import DataManager
from settings import Settings

logging.getLogger('tensorflow').disabled = True
logging.getLogger('numpy').disabled = True
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# Number of epochs of training. (One epoch uses all training material)
parser.add_argument('--epochs', dest='epochs', type=int, default=2)
# Batch size for single training inference
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10)
# The model architecture to train (currently only inception-v4 and resnet-101 are available)
parser.add_argument('--model', dest='model_architecture', default='inception', choices=['inception', 'resnet'])
# The name of the trained model (used for saving the training results)
parser.add_argument('--name', dest='model_name', default=datetime.datetime.now().strftime("%I_%M%p_on_%B_%d,_%Y"))
# The name of the dataset used for training. If available, the dataset will be downloaded automatically.
# isf-complete -> complete dataset, all signs, various lighting, precisely cropped
# isf-new -> also limited, better crop, but strange lighting (red shimmer)
# isf -> limited dataset with 4 signs, badly cropped.
# gtsrb -> real street signs, (Not all classes from CC present, even contains new classes)
# mnist -> standard hand written digits dataset
# cifar -> standard 32x32 image classification dataset (may be used for pre-training a model)
parser.add_argument('--dataset', dest='dataset_name', default='isf-complete',
                    choices=['isf', 'isf-new', 'isf-complete', 'gtsrb', 'mnist', 'cifar', 'cifar100'])
# Whether to augment the available training data at the start of the training
parser.add_argument('--augment', dest='augment_dataset', type=bool, default=False, choices=[True, False])
# For each real image, how many augmented images should be generated? (Program keeps track of original images via
# index file. If scalar is decreased after initial augmentation, augmented images will be deleted)
parser.add_argument('--augment_scalar', dest='augment_scalar', type=float, default=2.0)
# Path to settings file. (Settings will be restored from the given file)
parser.add_argument('--settings', dest='path_to_settings', type=str, default=None)
# Skip training in execution
parser.add_argument('--no-train', dest='train', action='store_false', default=True)
# save model after training
parser.add_argument('--save', dest='save', action='store_true', default=False)
# Export model (creates zip with settings and frozen model)
parser.add_argument('--export', dest='export', action='store_true', default=False)
# Execute trained model (shows images and their classifications)
parser.add_argument('--execute', dest='execute', action='store_true', default=False)

# Change only if you know what you are doing:
# Image height that neural net expects. (Images of differing sizes will be scaled)
parser.add_argument('--height', dest='height', default='auto')
# Image width that neural net expects. (Images of differing sizes will be scaled)
parser.add_argument('--width', dest='width', default='auto')
# Number of channels that the neural net expects. (If channels=1, conversion to grayscale is applied if necessary)
parser.add_argument('--channels', dest='channels', type=int, default=3)
# Dropout, only applied to dense layers (Dropout=percentage of neurons randomly omitted in training)
parser.add_argument('--dropout', dest='dropout_probability', type=float, default=0.2)
# Backbone layers won't be trainable
parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', default=False)

args = parser.parse_args()

train_path, test_path = util.get_necessary_dataset(args.dataset_name, Settings.data_path_from_root)

if args.augment_dataset:
    util.augment_data(scalar=args.augment_scalar, path_to_data=train_path, path_to_index=util.get_index(train_path),
                      balance=True)

if args.path_to_settings is None:
    settings = Settings({
        'height': args.height,
        'width': args.width,
        'channels': args.channels,
        'batch_size': args.batch_size,
        'train_data_dir': train_path,
        'test_data_dir': test_path,
        'model_name': args.model_name,
        'model_architecture': args.model_architecture,
        'dropout': args.dropout_probability,
        'epochs': args.epochs,
        'freeze_backbone': args.freeze_backbone
    })
    restore_model = None

else:
    settings = Settings(None, restore_from_path=args.path_to_settings)
    restore_model = settings.get_setting_by_name('model_save_path')

settings.assess(args)

print('Loading and preparing data..')
data_manager = DataManager(settings)

# Train model
if args.train:
    try:
        print('Training Model..')
        model = model_util.train(settings, data_manager, restore_model=restore_model)
    except KeyboardInterrupt:
        print('Stop training..')

# Save model
if args.save:
    os.makedirs(settings.get_save_path(), exist_ok=True)
    path = settings.get_save_path() + settings.get_setting_by_name('model_architecture') + '.h5'
    settings.update({'model_save_path': path}, write_to_file=True)
    print('Saving model to ' + path + ' (This can take a few minutes :/)')
    model.save_weights(path)

# Do inference
if args.execute:
    print('Evaluating Model..')
    test_data_generator = data_manager.yield_test_batch(settings.get_setting_by_name('batch_size'))
    for i in range(data_manager.get_batches_per_test_epoch()):
        images, labels = next(test_data_generator)
        model_util.evaluate_images_model_loaded(settings, model, images, labels=labels)
    cv2.destroyAllWindows()

# Export model to zip. Note that the model must have been saved or restored for this to work
if args.export:
    util.export_model_to_production(settings)
