import cv2
import numpy as np
import os
import sys
from ascii_graph import Pyasciigraph
import math

class DataManager:
    """
    This class reads and augments the data and ultimately provides
    train and test data to a model in batch-sized portions.
    """
    def __init__(self, settings):
        self.height = settings.get_setting_by_name('height')
        self.width = settings.get_setting_by_name('width')
        self.channels = settings.get_setting_by_name('channels')
        self.batch_size = settings.get_setting_by_name('batch_size')

        if not os.path.exists(settings.get_setting_by_name('train_data_dir')):
            print('train data not found.')
            classes_train = [None]
            self.train_data_generator = lambda: (yield None)
        else:
            print('collecting train data:')
            self._number_train_images, classes_train = self.read(settings.get_setting_by_name('train_data_dir'))
            # self.train_data_generator = self.read_yield(settings.get_setting_by_name('train_data_dir'))

        if not os.path.exists(settings.get_setting_by_name('test_data_dir')):
            print('test data not found.')
            classes_test = [None]
            self.test_data_generator = lambda: (yield None)
        else:
            print('collecting test data:')
            self._number_test_images, classes_test = self.read(settings.get_setting_by_name('test_data_dir'))
            # self.test_data_generator = self.read_yield(settings.get_setting_by_name('test_data_dir'))

        if not (len(classes_train) <= 1 or len(classes_test) <= 1):  # either test or train data was not found
            if len(classes_train) != len(classes_test):
                print("Warning: number of classes of train and test set don't match! This will fail if you use Keras!")

        settings.update({'num_classes': len(classes_train),
                         'class_names': classes_train,
                         'class_names_test': classes_test})

    def get_batches_per_train_epoch(self):
        """
        :return: the number of batches per training epoch
        """
        return math.ceil(self._number_train_images/self.batch_size)

    def get_batches_per_test_epoch(self):
        """
        :return: the number of batches per training epoch
        """
        return math.ceil(self._number_test_images/self.batch_size)

    def get_number_train_samples(self):
        """
        :return: the number of batches per training epoch
        """
        return self._number_train_images

    def get_number_test_samples(self):
        """
        :return: the number of batches per training epoch
        """
        return self._number_test_images

    def image_conversion(self, image):
        """
        Converts image to grayscale and resizes according to settings.
        :param image: the image to be converted.
        :return: the converted image as numpy array.
        """
        if len(np.shape(image)) < 3:
            image = np.expand_dims(image, 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, ) if self.channels == 1 else image
        image = cv2.resize(image, (self.height, self.width))
        return image

    @staticmethod
    def read(data_dir, print_distribution=True):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        :param data_dir: the directory containing a folder for each class which in turn contain the data.
        :param print_distribution: if true, a distribution of samples over the classes will be printed to console.
        :return: the images, labels and number of classes.
        """
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
                counter += 1
                image_counter += 1
            classes.append(subdirectories[i])
            distribution.append((subdirectories[i], counter))

        if print_distribution:
            graph = Pyasciigraph()
            for line in graph.graph('\nclass distribution:', distribution):
                print(line)

        print("images found in total: " + str(image_counter) + "\n")
        return image_counter, classes

    def read_yield(self, data_dir):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        :param data_dir: the directory containing a folder for each class which in turn contain the data.
        :param batch_size: size of each batch supplied by the generator function
        :return: the images, labels and number of classes.
        """
        subdirectories = list(os.walk(data_dir))[0][1]
        subdirectories = sorted(subdirectories)
        class_index = 0
        while True:
            path = os.path.join(data_dir, subdirectories[class_index])
            list_dir = os.listdir(path)
            for image_index in range(0, len(list_dir), self.batch_size):
                images = []
                labels = []
                for batch_index in range(self.batch_size):
                    if image_index + batch_index >= len(list_dir):
                        break
                    image = list_dir[image_index + batch_index]
                    if '.csv' in image:
                        continue
                    image = cv2.imread(os.path.join(path, image))
                    images.append(self.image_conversion(image))
                    label = np.zeros(len(subdirectories), np.float32)
                    label[class_index] = 1.0
                    labels.append(label)

                batch = (np.array(images), np.array(labels))
                print(" - class index: " + str(class_index)
                      + " image index: " + str(image_index)
                      + " batch size: " + str(len(batch[0])))
                yield batch

            class_index = (class_index + 1) % len(subdirectories)

