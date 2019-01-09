import os
import random
import sys

import cv2
import util
from PIL import Image
import PIL
import numpy as np
from math import pi


class Generator:

    """
    This class serves to generate artificial training images to enhance the training process
    TODO: simulate lighting
    TODO: normalize
    TODO: fix random bug where cropping messes up the image
    """
    def __init__(self, settings, random_colors=False, path_to_background=None, path_to_foreground=None):
        self.settings = settings
        if path_to_background is None or not os.path.exists(path_to_background):
            path_to_background = util.get_necessary_data('sliding_window', settings.data_path_from_root)
        if path_to_foreground is None or not os.path.exists(path_to_foreground):
            path_to_foreground = util.get_necessary_data('signs_clean', settings.data_path_from_root)
        self.foreground, self.class_names = util.read_any_data(path_to_foreground, imread_unchanged=True, return_filenames=True)
        self.background = util.read_any_data(path_to_background)
        self.color = True if settings.get_setting_by_name('channels') > 1 else False
        self.max_batches_per_epoch = settings.get_setting_by_name('maximum_artificial_batches_per_epoch')
        self.random_colors = random_colors
        self.batch_counter = 0

    def generate(self, n_samples):
        """
        Generates a specified number of training images from the background and foreground data.
        :param n_samples: number of samples to be generated
        :return: The samples as numpy array
        """
        samples_per_class = int(n_samples / len(self.foreground))

        samples = []
        labels = []
        for i in range(len(self.foreground)):
            foreground_image = self.foreground[i]
            for background_image in self.background:
                if len(samples) > samples_per_class * (i+1):
                    break
                sample = sample_image(background_image, foreground_image, random.uniform(0.4, 1.0), random.uniform(0, 0.5),
                                random.uniform(0.3, 0.6), random.uniform(0.3, 0.8), int(random.uniform(0, 60)), (random.randrange(-40, 40), random.randrange(-40, 40)))
                if not np.shape(sample)[0] > 0:
                    continue
                sample = cv2.resize(sample, (self.settings.get_setting_by_name('height'), self.settings.get_setting_by_name('width')))
                if self.random_colors:
                    sample = change_colors(sample, random.uniform(0, 1.0), random.uniform(0, 1.0), random.uniform(0, 1.0))
                if not self.color:
                    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                    sample = np.expand_dims(sample, 2)
                samples.append(sample)
                label = np.zeros([len(self.foreground)])
                label[i] = 1.0
                labels.append(label)

        if self.batch_counter > self.max_batches_per_epoch:
            self.batch_counter = 0
            return [], []
        else:
            self.batch_counter += 1
        return samples, labels

    def get_class_names(self):
        return self.class_names


def sample_image(bg, fg, x, y, scale, frame_scale, skew, crop_delta):
    """
    Creates a simulated image for training
    :param bg: Background image as nummpy array
    :param fg: Foreground image as numpy array
    :param y: where in the bg image should the fg image be placed (Value between 0 and 1)
    :param x: where in the bg image should the fg image be placed (Value between 0 and 1)
    :param brightness: change of brightness
    :param scale: scale of the fg image
    :param skew: tilt along the z-axis
    :param crop_delta: crop variation (given as 4-tuple: (change to y, change to x, change to height, change to width))
    :return: the sampled image
    """
    # get dimensions of crop frame and foreground image
    frame_height = int(np.shape(fg)[0] * frame_scale)
    frame_width = int(np.shape(fg)[1] * frame_scale)
    fg_height = int(np.shape(fg)[0] * scale)
    fg_width = int(np.shape(fg)[1] * scale)

    # scale foreground image
    fg = cv2.resize(fg, (fg_height, fg_width), interpolation=cv2.INTER_LINEAR)

    # rotate along z-axis
    fg = rotate_along_axis(fg, phi=skew, dz=18 + skew)

    # convert to PIL.Image
    fg = PIL.Image.fromarray(cv2.cvtColor(fg, cv2.COLOR_BGR2RGBA))
    bg = PIL.Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGBA))

    # get dimensions of background
    bg_width, bg_height = bg.size
    fg_width, fg_height = fg.size

    # determine absolute x and y position
    absolute_x = int(x*(bg_width - fg_width))
    absolute_y = int(y*(bg_height - fg_height))

    absolute_x_center = int(absolute_x + fg_width / 2)
    absolute_y_center = int(absolute_y + fg_height / 2)

    # paste on top of each other
    box = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
    box.paste(bg, (0, 0))
    box.paste(fg, (absolute_x, absolute_y), mask=fg)

    # convert back to numpy array
    img = cv2.cvtColor(np.array(box), cv2.COLOR_RGBA2BGR)

    # crop as per arguments given
    img = crop(img, (int(absolute_y_center - frame_height / 2), int(absolute_x_center - frame_width / 2)), (frame_height, frame_width), (crop_delta[0], crop_delta[1]))

    return img


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))


def deg_to_rad(deg):
    return deg * pi / 180.0


def rotate_along_axis(image, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    height = image.shape[0]
    width = image.shape[1]
    # Get radius of rotation along 3 axes
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(height ** 2 + width ** 2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal + dz

    # Get projection matrix
    mat = get_m(width, height, focal, rtheta, rphi, rgamma, dx, dy, dz)

    return cv2.warpPerspective(image.copy(), mat, (width, height))


def get_m(width, height, focal, theta, phi, gamma, dx, dy, dz):
    w = width
    h = height
    f = focal

    # Projection 2D -> 3D matrix
    a1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 1],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    ry = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])

    rz = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    r = np.dot(np.dot(rx, ry), rz)

    # Translation matrix
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0],
                   [0, f, h / 2, 0],
                   [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(t, np.dot(r, a1)))


def crop(image, roi_coordinates, roi_size, coordinate_delta):
    start_row = roi_coordinates[0] + coordinate_delta[0]
    end_row = start_row + roi_size[0]
    start_column = roi_coordinates[1] + coordinate_delta[1]
    end_column = start_column + roi_size[1]

    if start_row < 0:
        start_row = 0
    if start_column < 0:
        start_column = 0
    if end_row >= np.shape(image)[0]:
        end_row = np.shape(image)[0] - 1
    if end_column >= np.shape(image)[1]:
        end_column = np.shape(image)[1] - 1

    return image[start_row:end_row, start_column:end_column, :]


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_colors(img, r_change, g_change, b_change):
    # b, g, r, a = cv2.split(img)
    channels = cv2.split(img)
    b = channels[0]
    g = channels[1]
    r = channels[2]
    for i in range(len(b)):
        for j in range(len(b[i])):
            b[i][j] *= b_change
            g[i][j] *= g_change
            r[i][j] *= r_change
    channels[0] = b
    channels[1] = g
    channels[2] = r
    out = cv2.merge(channels)
    # out = cv2.merge([b, g, r, a])
    return out

