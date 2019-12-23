# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False, trainable=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias, trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1, trainable=trainable)

    branch_1 = conv2d_bn(input, 64, 1, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, trainable=trainable)

    branch_2 = conv2d_bn(input, 64, 1, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3, trainable=trainable)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1, trainable=trainable)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_a(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)

    branch_1 = conv2d_bn(input, 192, 1, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_b(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1, trainable=trainable)

    branch_1 = conv2d_bn(input, 192, 1, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1, trainable=trainable)

    branch_2 = conv2d_bn(input, 192, 1, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7, trainable=trainable)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', trainable=trainable)(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1, trainable=trainable)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_b(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1, trainable=trainable)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)

    branch_1 = conv2d_bn(input, 256, 1, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_c(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1, trainable=trainable)

    branch_1 = conv2d_bn(input, 384, 1, 1, trainable=trainable)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3, trainable=trainable)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1, trainable=trainable)
    branch_1 = merge([branch_10, branch_11], mode='concat', concat_axis=channel_axis)

    branch_2 = conv2d_bn(input, 384, 1, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1, trainable=trainable)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3, trainable=trainable)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3, trainable=trainable)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1, trainable=trainable)
    branch_2 = merge([branch_20, branch_21], mode='concat', concat_axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', trainable=trainable)(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1, trainable=trainable)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def inception_v4_base(input, trainable=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid', trainable=trainable)
    net = conv2d_bn(net, 64, 3, 3, trainable=trainable)

    branch_0 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1, trainable=trainable)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid', trainable=trainable)

    branch_1 = conv2d_bn(net, 64, 1, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1, trainable=trainable)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid', trainable=trainable)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2, 2), border_mode='valid', trainable=trainable)
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        net = block_inception_a(net, trainable=trainable)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net, trainable=trainable)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        net = block_inception_b(net, trainable=trainable)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net, trainable=trainable)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        net = block_inception_c(net, trainable=trainable)

    return net


def inception_v4_model(settings, load_imagenet_weights=True, trainable=False):
    '''
    Inception V4 Model for Keras

    Model Schema is based on
    https://github.com/kentsommer/keras-inceptionV4

    ImageNet Pretrained Weights 
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    '''

    height = settings.get_setting_by_name('height')
    width = settings.get_setting_by_name('width')
    channels = settings.get_setting_by_name('channels')
    num_classes = settings.get_setting_by_name('num_classes')
    dropout_keep_prob = settings.get_setting_by_name('dropout')

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((channels, height, width))
    else:
        inputs = Input((height, width, channels))

    # Make inception base
    net = inception_v4_base(inputs, trainable=trainable)

    # Final pooling and prediction

    # 8 x 8 x 1536
    net_old = AveragePooling2D((8, 8), border_mode='valid')(net)

    # 1 x 1 x 1536
    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    # 1536
    predictions = Dense(output_dim=1001, activation='softmax')(net_old)

    model = Model(inputs, predictions, name='inception_v4')

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = settings.INCEPTION_V4_WEIGHTS_PATH
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = settings.INCEPTION_V4_WEIGHTS_PATH

    if load_imagenet_weights:
        model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8, 8), border_mode='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(output_dim=num_classes, activation='softmax')(net_ft)

    model = Model(inputs, predictions_ft, name='inception_v4')

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
