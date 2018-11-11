import imageio
import numpy as np
import os
import sys
from skimage.transform import resize
from ascii_graph import Pyasciigraph

gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class DataManager:
    def __init__(self, settings, selection_mod=1):
        self.height = settings.get_setting_by_name('height')
        self.width = settings.get_setting_by_name('width')
        self.batch_size = settings.get_setting_by_name('batch_size')
        print('reading train data:')
        self.train_images, self.labels, num_classes_train = self.read(settings.get_setting_by_name('train_data_dir'), selection_mod)
        print('reading test data:')
        self.test_images, self.test_labels, num_classes_test = self.read(settings.get_setting_by_name('test_data_dir'), selection_mod)
        self.train_provider = RandomBatchProvider(self.train_images, self.labels, self.batch_size)
        self.test_provider = BatchProvider(self.train_images, self.labels, self.batch_size)
        if num_classes_train != num_classes_test:
            print("number of classes of train and test set don't match!")
            sys.exit(0)
        else:
            settings.update({'num_classes': num_classes_train})

    def image_conversion(self, img):
        img = gray(img)
        img = resize(img, (self.height, self.width), anti_aliasing=False)
        img = np.ndarray.astype(img, np.float32)  # convert to uint8
        return img

    def read(self, data_dir, selection_mod):
        """""
        reads all images from subdiretories 
        """""
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
                image = imageio.imread(os.path.join(path, image))
                images.append(self.image_conversion(image))
                label = np.zeros(len(subdirs), np.float32)
                label[i] = 1.0
                labels.append(label)
                counter += 1
            sys.stdout.write('\rreading files from subdirectory ' + str(i+1) + '/' + str(len(subdirs)))
            sys.stdout.flush()
            distribution.append((subdirs[i], counter))

        graph = Pyasciigraph()
        for line in graph.graph('\nclass distribution:', distribution):
            print(line)
        print(np.shape(images), ' ', np.shape(labels))
        print("total number of images: " + str(len(images)))
        images = np.array(images)
        labels = np.array(labels)
        images = np.reshape(images, [images.shape[0], images.shape[1] * images.shape[2]])
        if len(images) < self.batch_size:
            print('fewer images than a single batch size available!')
            sys.exit()
        return images, labels, len(subdirs)

    def next_batch(self):
        return self.train_provider.next_batch()

    def next_test_batch(self):
        return self.test_provider.next_batch()

    @staticmethod
    def get_num_videos(settings):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(settings.get_setting_by_name('train_data_dir')):
            f.extend(filenames)
        return len(f)


class BatchProvider:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.iteration = 0
        self.reset()

    def next_batch(self):
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

    def reset(self):
        self.iteration = 0


class RandomBatchProvider:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.iteration = 0
        self.reset()

    def next_batch(self):
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

    def reset(self):
        self.iteration = 0
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]
