import cv2
from abc import abstractmethod
import numpy as np
import os
import sys
from ascii_graph import Pyasciigraph


class DataManager:
    """
    This class reads and augments the data and ultimately provides
    train and test data to a model in batch-sized portions.
    """
    def __init__(self, settings, selection_mod=1):
        self.height = settings.get_setting_by_name('height')
        self.width = settings.get_setting_by_name('width')
        self.batch_size = settings.get_setting_by_name('batch_size')
        print('reading train data:')
        self.train_images, self.labels, num_classes_train = self.read(settings.get_setting_by_name('train_data_dir'), settings.get_setting_by_name('channels'))
        print('reading test data:')
        self.test_images, self.test_labels, num_classes_test = self.read(settings.get_setting_by_name('test_data_dir'), settings.get_setting_by_name('channels'))
        self.train_provider = RandomBatchProvider(self.train_images, self.labels, self.batch_size)
        self.test_provider = OrderedBatchProvider(self.test_images, self.test_labels, self.batch_size)
        if num_classes_train != num_classes_test:
            print("number of classes of train and test set don't match!")
            sys.exit(0)
        else:
            settings.update({'num_classes': num_classes_train})

    def image_conversion(self, image, channels):
        """
        Converts image to grayscale and resizes according to settings.
        :param image: the image to be converted.
        :return: the converted image as numpy array.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, ) if channels == 1 else image
        image = cv2.resize(image, (self.height, self.width))
        image = np.ndarray.astype(image, np.float32)
        return image

    def read(self, data_dir, channels, print_distribution=True):
        """
        Reads all images from subdirectories and creates corresponding labels.
        Optionally, a sample distribution over the classes will be printed.
        :param data_dir: the directory containing a folder for each class which in turn contain the data.
        :param print_distribution: if true, a distribution of samples over the classes will be printed to console.
        :param channels: number of channels for each image (has to be smaller or equal to channels of raw data)
        :return: the images, labels and number of classes.
        """
        images = []
        labels = []

        distribution = []

        subdirs = list(os.walk(data_dir))[0][1]
        for i in range(len(subdirs)):
            path = os.path.join(data_dir, subdirs[i])
            list_dir = os.listdir(path)
            counter = 0
            for image in list_dir:
                if '.csv' in image:
                    continue
                image = cv2.imread(os.path.join(path, image))
                if len(np.shape(image)) < 3:
                    image = np.expand_dims(image, 2)
                images.append(self.image_conversion(image, channels))
                label = np.zeros(len(subdirs), np.float32)
                label[i] = 1.0
                labels.append(label)
                counter += 1
            sys.stdout.write('\rreading files from subdirectory ' + str(i+1) + '/' + str(len(subdirs)))
            sys.stdout.flush()
            distribution.append((subdirs[i], counter))

        if print_distribution:
            graph = Pyasciigraph()
            for line in graph.graph('\nclass distribution:', distribution):
                print(line)
        print("total number of images: " + str(len(images)))
        print(np.shape(images))
        images = np.array(images)
        labels = np.array(labels)
        images = np.reshape(images, [images.shape[0], images.shape[1] * images.shape[2] * channels])
        if len(images) < self.batch_size:
            print('fewer images than a single batch size available!')
            sys.exit()
        return images, labels, len(subdirs)

    def next_batch(self):
        """
        Returns next train batch.
        :return: images and labels as numpy arrays. First dimension is equal to batch_size
        """
        return self.train_provider.next_batch()

    def next_test_batch(self):
        """
        Returns next test batch.
        :return: images and labels as numpy arrays. First dimension is equal to batch_size
        """
        return self.test_provider.next_batch()

    def batches_per_epoch(self):
        """
        :return: the number of batches per epoch
        """
        return int(len(self.train_images)/self.batch_size)


class BatchProvider:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.iteration = 0
        self.reset()

    def next_batch(self):
        """
        Returns next batch of size batch_size until no more data is available.
        If less data than a single batch_size remains, a single smaller batch will be returned.
        After this an empty batch will be returned to signal the end of the epoch and the dataset will be reset via the reset() function.
        :return: The next batch as specified above.
        """
        if self.iteration == -1:
            self.reset()
            return [], []
        if (self.iteration + 1) * self.batch_size > len(self.data) - 1:
            batch_x = self.data[self.iteration * self.batch_size:]
            batch_y = self.labels[self.iteration * self.batch_size:]
            self.iteration = -1
            return batch_x, batch_y
        else:
            batch_x = self.data[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            batch_y = self.labels[self.iteration * self.batch_size:(self.iteration + 1) * self.batch_size]
            self.iteration += 1
            return batch_x, batch_y

    @abstractmethod
    def reset(self):
        self.iteration = 0


class OrderedBatchProvider(BatchProvider):
    """
    Provides data and labels in the order they were first read.
    This is suitable for testing and validation, because order is irrelevant in these cases.
    """
    def reset(self):
        self.iteration = 0


class RandomBatchProvider(BatchProvider):
    """
    Provides data and labels in a random order.
    This provider should be used for the train data to keep the model balanced.
    """
    def reset(self):
        """
        Resets the data by shuffling data and labels jointly
        :return:
        """
        self.iteration = 0
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]
