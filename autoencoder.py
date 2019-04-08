import os

import tensorflow as tf
import numpy as np


def encoder(x, dropout_probability, is_training, settings):
    """
    This method builds the encoder part of the autoencoder.
    :param x: The input to be processed by the encoder. Expected to be of size [None, image_height, image_width, image_channels]
    :param dropout_probability: placeholder for dropout probability so that it can be utilized dynamically
    :param is_training:  A tensor containing a single boolean. Currently used for batch normalization
    :param settings: The settings determining the construction of the model.
    :return: The output produced by the encoder part of the autoencoder.
    """
    print('Encoder:')
    conv_filters = settings.get_setting_by_name('conv_filters')
    pooling_after_conv = settings.get_setting_by_name('pooling_after_conv')
    kernel_sizes = settings.get_setting_by_name('conv_kernels')
    fc_hidden_units = settings.get_setting_by_name('fc_hidden')
    temp_width = settings.get_setting_by_name('width')
    temp_height = settings.get_setting_by_name('height')
    temp_channels = settings.get_setting_by_name('channels')
    tensor = tf.reshape(x, [-1, temp_height, temp_width, temp_channels])

    print('input: ' + str(temp_width) + ' x ' + str(temp_height) + ' x  ' + str(temp_channels))

    with tf.variable_scope('cnn'):

        for i in range(len(conv_filters)):

            if settings.get_setting_by_name('batch_norm'):
                tensor = tf.layers.batch_normalization(tensor, training=is_training, center=True, scale=True)
                print('batch norm')

            tensor = tf.layers.conv2d(tensor, conv_filters[i], kernel_sizes[i], padding='SAME', activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            temp_channels = conv_filters[i]
            description = 'conv: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(conv_filters[i])
            print(description)
            if pooling_after_conv[i]:
                tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                temp_height = int(temp_height / 2)
                temp_width = int(temp_width / 2)
                print('pool: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(temp_channels))

    tensor = tf.reshape(tensor, shape=[-1, temp_height * temp_width * temp_channels])
    print('resize: ' + str(temp_width) + '*' + str(temp_height) + '*' + str(temp_channels))

    with tf.variable_scope('dnn'):
        for i in range(len(fc_hidden_units)):
            fc = tf.layers.dense(tensor, fc_hidden_units[i], activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            tensor = tf.nn.dropout(fc, dropout_probability)
            description = 'fc: ' + str(fc_hidden_units[i])
            print(description)

    # with tf.variable_scope('out'):
    #     encoding = tf.layers.dense(tensor, settings.get_setting_by_name('encoding_size'), activation=tf.nn.softmax,
    #                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    #                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    #     print('out: ' + str(settings.get_setting_by_name('encoding_size')))

    encoding = tensor

    return encoding


def decoder(encoding, dropout_probability, is_training, settings):
    """
    This method builds the decoder part of the autoencoder.
    The decoder is essentially the inverse operation of the encoder.
    We use subpixel and bilinear upsampling as the inverse of the pooling operation.
    :param encoding: The input to be processed by the decoder. Expected to be of size [encoding_size]
    :param dropout_probability: placeholder for dropout probability so that it can be utilized dynamically
    :param is_training:  A tensor containing a single boolean. Currently used for batch normalization
    :param settings: The settings determining the construction of the model.
    :return: The output produced by the decoder part of the autoencoder.
    """
    print('Decoder:')
    conv_filters = settings.get_setting_by_name('conv_filters')[::-1]
    pooling_after_conv = settings.get_setting_by_name('pooling_after_conv')[::-1]
    kernel_sizes = settings.get_setting_by_name('conv_kernels')[::-1]
    fc_hidden_units = settings.get_setting_by_name('fc_hidden')[::-1]
    upsampling_type = settings.get_setting_by_name('upsampling_type')
    n_pooling_operations = np.sum(pooling_after_conv)
    temp_width = int(settings.get_setting_by_name('width') / 2 ** n_pooling_operations)
    temp_height = int(settings.get_setting_by_name('height') / 2 ** n_pooling_operations)
    temp_channels = conv_filters[0]

    tensor = encoding

    with tf.variable_scope('dnn'):
        for i in range(len(fc_hidden_units)):
            fc = tf.layers.dense(tensor, fc_hidden_units[i], activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            tensor = tf.nn.dropout(fc, dropout_probability)
            description = 'fc: ' + str(fc_hidden_units[i])
            print(description)

    tensor = tf.reshape(tensor, shape=[-1, temp_height, temp_width, temp_channels])
    print('resize: ' + str(temp_width) + 'x' + str(temp_height) + 'x' + str(temp_channels))

    with tf.variable_scope('cnn'):

        for i in range(len(conv_filters)):

            if settings.get_setting_by_name('batch_norm'):
                tensor = tf.layers.batch_normalization(tensor, training=is_training, center=True, scale=True)
                print('batch norm')

            tensor = tf.layers.conv2d(tensor, conv_filters[i], kernel_sizes[i], padding='SAME', activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            temp_channels = conv_filters[i]
            description = 'conv: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(conv_filters[i])
            print(description)
            if pooling_after_conv[i]:
                temp_height = int(temp_height * 2)
                temp_width = int(temp_width * 2)
                if upsampling_type == 'bilinear':
                    description = 'bilinear'
                    size = [int(tensor.shape[1] * 2), int(tensor.shape[2] * 2)]
                    tensor = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
                elif upsampling_type == 'subpixel':
                    description = 'subpixel'
                    upsample_filters = tf.layers.conv2d(input, temp_channels * 2 * 2, (2, 2), padding='same', activation=tf.nn.relu,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                        bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                    tensor = tf.depth_to_space(upsample_filters, 2)

                print(description + '-upsampling: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(temp_channels))

    with tf.variable_scope('out'):
        if settings.get_setting_by_name('batch_norm'):
            tensor = tf.layers.batch_normalization(tensor, training=is_training, center=True, scale=True)
            print('batch norm')

        temp_channels = settings.get_setting_by_name('channels')
        output = tf.layers.conv2d(tensor, temp_channels, 3, padding='SAME', activation=tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                  bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                  name=settings.get_setting_by_name('output_node_name'))
        print('output: ', temp_width, ' x ', temp_height, ' ', temp_channels)

    return output


def auto_encoder(x, y, dropout_probability, is_training, settings):
    """
    This function builds the autoencoder tensorflow
    graph, consisting of encoder -> decoder -> loss
    :param x: Input to be processed by the autoencoder
    :param y: Desired output/label
    :param dropout_probability: tf variable representing probability for dropout (Passed with value 1 on test inference)
    :param is_training: Currently used for batch norm
    :param settings: Settings, determining how to build the model
    :return: Actual output of the autoencoder as well as the loss when comparing to y
    """
    print('Building auto encoder:')

    with tf.variable_scope('encoder'):
        encoding = encoder(x, dropout_probability, is_training, settings)

    with tf.variable_scope('decoder'):
        output = decoder(encoding, dropout_probability, is_training, settings)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(tf.layers.flatten(output), y)))

    return output, loss


def train(settings, data_manager, n_epochs=400, restore_type='auto', show_test=False):
    """
    Train a model.
    While the model keeps improving it will be saved to disk.
    Once a model has been saved it can be retrieved in various ways given through restore_type and restore_argument.

    :param settings: the settings providing all the necessary information for building the model and gathering the input data.
    :param n_epochs: the number of epochs for which the model will be trained. One epoch utilizes all training material exactly once.
    :param restore_type: restore model and continue train. Either
                        'auto': for automatic continuation
                        'path': for reading save from path given in restore_argument
                        'by_name': for restoring from save from settings.save_path dir with name restore_argument
                        'transfer': restores only the convolutional layers and retrains the dense layers, restores from path given in restore_argument
    :param show_test: show classified images
    :param restore_argument: depending on the restore type, restore data must provide the according information - either the name or path of a saved model
    """
    # np.random.seed(settings.get_setting_by_name('seed'))  # doesn't work currently
    # tf.set_random_seed(settings.get_setting_by_name('seed'))  # doesn't work currently
    x = tf.placeholder(tf.float32, [None, settings.get_setting_by_name('width'), settings.get_setting_by_name('height'),
                                    settings.get_setting_by_name('channels')], name=settings.get_setting_by_name('input_node_name'))
    y = tf.placeholder(tf.float32, [None, settings.get_setting_by_name('num_classes')])

    dropout_probability = tf.placeholder_with_default(1.0, shape=())
    is_training = tf.placeholder_with_default(False, shape=())

    prediction, loss = auto_encoder(x, y, dropout_probability, is_training, settings)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=settings.get_setting_by_name('learning_rate'),
                                               global_step=global_step, decay_steps=data_manager.batches_per_epoch(),
                                               decay_rate=settings.get_setting_by_name('learning_rate_decay'))

    # extra operations for batch norm
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # make dependency for train step
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar("cost", loss)
    merged_summary_op = tf.summary.merge_all()

    # config = tf.ConfigProto(device_count={'GPU': 0}) # train without gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(settings.get_logs_path(), graph=tf.get_default_graph())
        if settings.get_setting_by_name('input_checkpoint') is not None:
            if restore_type == 'transfer':
                print('WARNING: transfer learning not implemented for autoencoder. No parts will be locked')
                training_lock = settings.get_setting_by_name('training_lock')
                if 'cnn' in training_lock:
                    cnn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn'))
                    cnn_saver.restore(sess, settings.get_setting_by_name('input_checkpoint'))
                    print('CNN restored from ', settings.get_setting_by_name('input_checkpoint'), ' and LOCKED!')
                if 'dnn' in training_lock:
                    if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn')) <= 0:
                        print('No DNN to be restored.')
                    else:
                        dnn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn'))
                        dnn_saver.restore(sess, settings.get_setting_by_name('input_checkpoint'))
                        print('DNN restored from ', settings.get_setting_by_name('input_checkpoint'), ' and LOCKED!')
            else:
                saver.restore(sess, settings.get_setting_by_name('input_checkpoint'))

        min_cost = -1
        no_improvement_counter = 0
        test_loss = 0
        min_test_loss = 0
        for epoch in range(n_epochs):

            # ===== TRAIN =====
            loss_sum = 0
            batch_x, batch_y = data_manager.next_batch()

            while len(batch_x) != 0:
                _prediction, _, c, summary = sess.run([prediction, optimizer, loss, merged_summary_op],
                                                      feed_dict={x: batch_x, y: batch_y, is_training: True,
                                                                 dropout_probability: settings.get_setting_by_name('dropout')})
                loss_sum += c
                batch_x, batch_y = data_manager.next_batch()
            loss_avg = loss_sum / data_manager.batches_per_epoch()
            train_accuracy_print = 'train accuracy: ' + str(round(loss_avg, 2)) + '%'

            # ===== TEST =====
            test_loss = 0
            test_batch_x, test_batch_y = data_manager.next_test_batch()

            while len(test_batch_x) != 0:
                test_prediction, c = sess.run([prediction, loss], feed_dict={x: test_batch_x})
                test_loss += c
                test_batch_x, test_batch_y = data_manager.next_test_batch()
            test_accuracy = 100 * test_loss / data_manager.batches_per_test()
            test_accuracy_print = 'test accuracy: ' + str(round(test_accuracy, 2)) + '%'

            train_info = 'Epoch ' + str(epoch + 1) + ' / ' + str(n_epochs) + ' cost: ' + str(round(loss_avg, 3))

            # === PRINT EVAL ===
            if min_cost > loss_sum or min_cost == -1:
                min_cost = loss_sum
                no_improvement_counter = 0
                train_info += ' :)'
            else:
                no_improvement_counter += 1
                if no_improvement_counter > 12:
                    train_info += ' :('
                else:
                    train_info += ' :|'

            print(train_info, ' ', train_accuracy_print, ' ', test_accuracy_print)

            # === SAVE IF IMPROVEMENT ===
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                save_dir = settings.get_save_path()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + 'epoch' + str(epoch+1) + '.ckpt'
                saver.save(sess, save_path)
                settings.update({'model_save_path': save_path, 'input_checkpoint': save_path})
                print('Model saved at ', save_path)
            elif test_loss == 0:  # save if meaningful testing is not possible (e.g. no test data available)
                save_dir = settings.get_save_path()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + 'epoch' + str(epoch + 1) + '.ckpt'
                saver.save(sess, save_path)
                settings.update({'model_save_path': save_path, 'input_checkpoint': save_path})
                print('Model saved at ', save_path)

        print('Optimization Finished')
        settings.update({'input_checkpoint': settings.get_setting_by_name('model_save_path')})
        sess.close()
