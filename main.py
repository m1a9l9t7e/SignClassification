import argparse
import datetime
import warnings
import os
import cv2
import numpy as np
from models.resnet_101 import resnet101_model as build_resnet
from models.inception_v4 import inception_v4_model as build_inception
import util
from data_generator import DataManager
from settings import Settings

warnings.filterwarnings("ignore")


def train(settings, data_manager, restore_model=None):
    # Construct model and load imagenet weights
    architecture = settings.get_setting_by_name('model_architecture')
    if architecture == 'inception':
        model = build_inception(settings, load_imagenet_weights=(restore_model is None))
    elif architecture == 'resnet':
        model = build_resnet(settings, load_imagenet_weights=(restore_model is None))
    else:
        print('Error: model architecture ' + architecture + ' is currently not supported')
        return None

    # Continue training if custom restore weights are specified
    if restore_model is not None:
        model.load_weights(restore_model, by_name=True)

    # Start Fine-tuning
    model.fit_generator(data_manager.yield_train_batch(settings.get_setting_by_name('batch_size')),
                        nb_epoch=settings.get_setting_by_name('epoch'),
                        samples_per_epoch=data_manager.get_number_train_samples(),
                        verbose=1,
                        validation_data=data_manager.yield_test_batch(settings.get_setting_by_name('batch_size')),
                        nb_val_samples=data_manager.get_number_test_samples(),
                        max_q_size=10
                        )

    return model


def evaluate_images_model_loaded(settings, model, images, labels=None, show=True):
    predictions_valid = model.predict(images, batch_size=settings.get_setting_by_name('batch_size'), verbose=1)
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


def evaluate_images(settings, path_to_model, images, labels=None):
    architecture = settings.get_setting_by_name('model_architecture')
    if architecture == 'inception':
        model = build_inception(settings, load_imagenet_weights=False)
    elif architecture == 'resnet':
        model = build_resnet(settings, load_imagenet_weights=False)
    else:
        print('Error: model architecture ' + architecture + ' is currently not supported')
        return None
    model.load_weights(path_to_model, by_name=True)
    evaluate_images_model_loaded(settings, model, images, settings, labels)


parser = argparse.ArgumentParser()
# Number of epochs of training. (One epoch uses all training material)
parser.add_argument('--epoch', dest='epoch', type=int, default=5)
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
        'epoch': args.epoch
    })

else:
    settings = Settings(None, restore_from_path=args.path_to_settings)

# check if model was downloaded
settings.assess(args)

print('Loading and preparing data..')
data_manager = DataManager(settings)

# Train model
if args.train:
    try:
        print('Training Model..')
        model = train(settings, data_manager)
    except KeyboardInterrupt:
        print('Stop training..')

# Save model
if args.save:
    os.makedirs(settings.get_save_path(), exist_ok=True)
    path = settings.get_save_path() + settings.get_setting_by_name('model_architecture') + '.h5'
    print('Saving model to ' + path + ' (This can take a few minutes :/)')
    model.save_weights(path)

# Do inference
if args.execute:
    print('Evaluating Model..')
    test_data_generator = data_manager.read_yield(settings.get_setting_by_name('test_data_dir'))
    for i in range(data_manager.get_batches_per_test_epoch()):
        images, labels = next(test_data_generator)
        evaluate_images_model_loaded(settings, model, images, labels=labels)
    cv2.destroyAllWindows()

# if args.export:
#     TODO
