import cv2
import numpy as np
import os
import sys
from ascii_graph import Pyasciigraph
import math
from util import transform


class DataManager:
    """
    This class reads and augments the data and ultimately provides
    train and test data to a model in batch-sized portions.
    """
    training_data = []
    test_data = []

    def __init__(self, settings):
        self.settings = settings

        if not os.path.exists(settings.get_setting_by_name('train_data_dir')):
            print('train data not found.')
            self.training_data = []
            self.classes_train = []
        else:
            print('collecting train data:')
            self.training_data, self.classes_train = self.read(settings.get_setting_by_name('train_data_dir'))

        if not os.path.exists(settings.get_setting_by_name('test_data_dir')):
            print('test data not found.')
            self.test_data = []
            self.classes_test = []
        else:
            print('collecting test data:')
            self.test_data, self.classes_test = self.read(settings.get_setting_by_name('test_data_dir'))

        if not (len(self.classes_train) <= 1 or len(self.classes_test) <= 1):  # either test or train data was not found
            if len(self.classes_train) != len(self.classes_test):
                print("Warning: number of classes of train and test set don't match! This will fail if you use Keras!")
                print("Aborting..")
                sys.exit(0)

        settings.update({'num_classes': len(self.classes_train),
                         'class_names': self.classes_train,
                         'class_names_test': self.classes_test})

    def get_batches_per_train_epoch(self):
        """
        :return: the number of batches per training epoch
        """
        return math.ceil(len(self.training_data[0])/self.settings.get_setting_by_name('batch_size'))

    def get_batches_per_test_epoch(self):
        """
        :return: the number of batches per test epoch
        """
        return math.ceil(len(self.test_data[0])/self.settings.get_setting_by_name('batch_size'))

    def get_number_train_samples(self):
        """
        :return: the number of training samples
        """
        return len(self.training_data[0])

    def get_number_test_samples(self):
        """
        :return: the number of test samples
        """
        return len(self.test_data[0])

    @staticmethod
    def read(data_dir, print_distribution=True):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        :param data_dir: the directory containing a folder for each class which in turn contain the data.
        :param print_distribution: if true, a distribution of samples over the classes will be printed to console.
        :return: the images, labels and number of classes.
        """
        filenames = []
        labels = []

        distribution = []
        classes = []
        image_counter = 0

        subdirectories = list(os.walk(data_dir))[0][1]
        subdirectories = sorted(subdirectories)
        for i in range(len(subdirectories)):
            path = os.path.join(data_dir, subdirectories[i])
            list_dir = os.listdir(path)
            counter = 0
            for image in list_dir:
                if '.csv' in image:
                    continue
                filenames.append(os.path.join(path, image))
                labels.append(i)
                counter += 1
                image_counter += 1
            classes.append(subdirectories[i])
            distribution.append((subdirectories[i], counter))

        if print_distribution:
            graph = Pyasciigraph()
            for line in graph.graph('\nclass distribution:', distribution):
                print(line)

        print("images found in total: " + str(image_counter) + "\n")
        return (np.array(filenames), np.array(labels)), classes

    def yield_train_batch(self, batch_size):
        files, label_indices = self.training_data
        index = 0

        while True:
            if index == 0:
                permutation = np.arange(self.get_number_train_samples())
                np.random.shuffle(permutation)
                files = files[permutation]
                label_indices = label_indices[permutation]

            images = []
            labels = []
            for i in range(batch_size):
                if index + i >= self.get_number_train_samples():
                    break
                image = cv2.imread(files[index + i])
                images.append(transform(image, self.settings))
                label = self.encode_one_hot(len(self.classes_train), label_indices[index + i])
                labels.append(label)

            yield (np.array(images), np.array(labels))
            index = (index + batch_size) % self.get_number_train_samples()

    def yield_test_batch(self, batch_size):
        files, label_indices = self.test_data
        index = 0

        while True:
            images = []
            labels = []
            for i in range(batch_size):
                if index + i >= self.get_number_test_samples():
                    break
                image = cv2.imread(files[index + i])
                images.append(transform(image, self.settings))
                label = self.encode_one_hot(len(self.classes_test), label_indices[index + i])
                labels.append(label)

            yield (np.array(images), np.array(labels))
            index = (index + batch_size) % self.get_number_test_samples()

    @staticmethod
    def encode_one_hot(nb_classes, class_index):
        label = np.zeros(nb_classes, np.float32)
        label[class_index] = 1.0
        return label
