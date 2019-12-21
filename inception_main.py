import argparse
import datetime
import sys
import warnings
import os
import cv2
import numpy as np
# from refinement.resnet_101 import resnet101_model as build_model
from inception_v4 import inception_v4_model as build_model
import util
from data import DataManager
from settings import Settings

warnings.filterwarnings("ignore")


def train(settings, batch_size, epochs, X_train, Y_train, X_valid, Y_valid, restore_model=None, save_model=True, save_name="inception-v4"):
    # Construct model and load imagenet weights
    model = build_model(settings.get_setting_by_name('height'), settings.get_setting_by_name('width'),
                        settings.get_setting_by_name('channels'), settings.get_setting_by_name('num_classes'),
                        load_weights=(restore_model is None))

    # Continue training if custom restore weights are specified
    if restore_model is not None:
        model.load_weights(restore_model, by_name=True)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    if save_model:
        os.makedirs(settings.get_save_path(), exist_ok=True)
        path = settings.get_save_path() + save_name + '.h5'
        print('Saving model to ' + path + ' (This can take a few minutes :/)')
        model.save_weights(path)

    return model


def evaluate_images_model_loaded(settings, model, images, batch_size, labels=None, show=True):
    predictions_valid = model.predict(images, batch_size=batch_size, verbose=1)
    predictions = []

    for i in range(len(images)):
        predicted_class = settings.get_setting_by_name('class_names')[np.argmax(predictions_valid[i])]
        predictions.append(predicted_class)

        if not show:
            continue

        print(str(i), ': ', predicted_class)

        if labels is not None:
            # print(predicted_class, ' vs ', settings.get_setting_by_name('class_names')[np.argmax(labels[i])])
            prediction_correct = predicted_class == settings.get_setting_by_name('class_names')[np.argmax(labels[i])]
            image = cv2.putText(images[i], predicted_class, (0, settings.get_setting_by_name('height') // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if prediction_correct else (0, 0, 255),
                                thickness=2)
        else:
            image = cv2.putText(images[i], predicted_class, (0, settings.get_setting_by_name('height') // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        # if labels is not None and not prediction_correct:
        cv2.imshow('results', image)
        cv2.waitKey(0)

    return predictions


def evaluate_images(images, path_to_model, settings, labels=None):

    model = build_model(settings.get_setting_by_name('height'), settings.get_setting_by_name('width'),
                        settings.get_setting_by_name('channels'), settings.get_setting_by_name('num_classes'),
                        load_weights=False)
    model.load_weights(path_to_model, by_name=True)
    evaluate_images_model_loaded(settings, model, images, settings, labels)


def make_predictions(path_to_images, path_to_model, settings):
    images = util.read_any_data(path_to_images, settings=settings)
    evaluate_images(images, path_to_model, settings.get_setting_by_name('batch_size'), settings)


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
parser.add_argument('--model', dest='model_name', default=datetime.datetime.now().strftime("%I_%M%p_on_%B_%d,_%Y"))
# The name of the dataset used for training. If available, the dataset will be downloaded automatically.
# isf -> limited dataset with 4 signs, badly cropped.
# isf-new -> also limited, better crop, but strange lighting (red shimmer)
# isf-complete -> complete dataset, all signs, various lighting, precisely cropped
# gtsrb -> real street signs, (Not all classes from CC present, even contains new classes)
# mnist -> standard hand written digits dataset
# cifar10 -> standard 32x32 image classification dataset (may be used for pre-training a model)
parser.add_argument('--dataset', dest='dataset_name', default='isf', choices=['isf', 'isf-new', 'isf-complete', 'gtsrb', 'mnist', 'cifar', 'cifar100'])
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
# If 'transfer' is used with (--restore), this argument decides which parts of the model will be restored and locked
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

args = parser.parse_args()
train_path, test_path = util.get_necessary_dataset(args.dataset_name, Settings.data_path_from_root)

if args.augment_dataset:
    util.augment_data(scalar=args.augment_scalar, path_to_data=train_path, path_to_index=util.get_index(train_path), balance=True)

if args.path_to_settings is None:
    settings = Settings({
        'height': args.height,
        'width': args.width,
        'channels': args.channels,
        'batch_size': 999999,  # This is a workaround, don't change
        'train_data_dir': train_path,
        'test_data_dir': test_path,
        'model_name': args.model_name,
    })

else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

# settings.assess(args)
print('Loading and preparing data..')
data_manager = DataManager(settings)

num_classes = int(settings.get_setting_by_name('num_classes'))

X_train, Y_train = data_manager.next_batch()
X_valid, Y_valid = data_manager.next_test_batch()

if args.train:
    try:
        print('Training Model..')
        model = train(settings, args.batch_size, args.epoch, X_train, Y_train, X_valid, Y_valid)
    except KeyboardInterrupt:
        print('Stop training..')


print('Evaluating Model..')
evaluate_images_model_loaded(settings, model, X_valid, args.batch_size, labels=Y_valid)
# evaluate_images(X_valid, rf_const.MODEL_SAVE_PATH, rf_const.BATCH_SIZE, settings, labels=Y_valid)


# if args.freeze:
#     tf_util.freeze_graph(settings)

# if args.export:
#     util.export_model_to_production(settings)

# if args.execute:
#     tf_util.execute_frozen_model(settings, data_manager.test_provider)
