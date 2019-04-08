import cv2
import os
import numpy as np
import tensorflow as tf


def model(x, y, dropout_probability, is_training, settings):
    """
    This method builds the classification model
    :param x: The input to be processed by the classifier. Expected to be of size [None, image_height, image_width, image_channels]
    :param y: The label corresponding to input x
    :param dropout_probability: placeholder for dropout probability so that it can be utilized dynamically
    :param is_training:  A tensor containing a single boolean. Currently used for batch normalization
    :param settings: The settings determining the construction of the model.
    :return: The output produced by the model as well as the loss.
    """
    print('Building model:')
    conv_filters = settings.get_setting_by_name('conv_filters')
    pooling_after_conv = settings.get_setting_by_name('pooling_after_conv')
    kernel_sizes = settings.get_setting_by_name('conv_kernels')
    fc_hidden_units = settings.get_setting_by_name('fc_hidden')
    training_lock = settings.get_setting_by_name('training_lock')
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
                                      bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=('cnn' not in training_lock))
            temp_channels = conv_filters[i]
            description = 'conv: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(conv_filters[i])
            if 'cnn' in training_lock:
                description += ' locked!'
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
            tensor = tf.layers.dense(tensor, fc_hidden_units[i], activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=('dnn' not in training_lock))
            tensor = tf.nn.dropout(tensor, dropout_probability)
            description = 'fc: ' + str(fc_hidden_units[i])
            if 'dnn' in training_lock:
                description += ' locked!'
            print(description)

    with tf.variable_scope('out-dnn'):
        if temp_width * temp_height * temp_channels == settings.get_setting_by_name('num_classes'):
            output = tensor
            print('fully convolutional resize')
        else:
            output = tf.layers.dense(tensor, settings.get_setting_by_name('num_classes'),
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                     bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    output = tf.nn.softmax(output, name=settings.get_setting_by_name('output_node_name'))
    print('output: ' + str(settings.get_setting_by_name('num_classes')))

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(output, y)))

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

    prediction, loss = model(x, y, dropout_probability, is_training, settings)

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

        class_names_test = settings.get_setting_by_name('class_names_test')
        class_names_model = settings.get_setting_by_name('class_names')
        min_cost = -1
        no_improvement_counter = 0
        test_accuracy = 0
        max_accuracy = 0
        for epoch in range(n_epochs):

            # ===== TRAIN =====
            loss_sum = 0
            correct = 0
            wrong = 0
            batch_x, batch_y = data_manager.next_batch()

            while len(batch_x) != 0:
                _prediction, _, c, summary = sess.run([prediction, optimizer, loss, merged_summary_op],
                                                      feed_dict={x: batch_x, y: batch_y, is_training: True,
                                                                 dropout_probability: settings.get_setting_by_name('dropout')})
                loss_sum += c
                for i in range(np.shape(batch_y)[0]):
                    if np.argmax(batch_y[i]) == np.argmax(_prediction[i]):
                        correct += 1
                    else:
                        wrong += 1
                batch_x, batch_y = data_manager.next_batch()
            train_accuracy = 100 * correct / (correct + wrong)
            train_accuracy_print = 'train accuracy: ' + str(round(train_accuracy, 2)) + '%'
            loss_avg = loss_sum / (correct + wrong)

            # ===== TEST =====
            correct = 0
            wrong = 0
            test_batch_x, test_batch_y = data_manager.next_test_batch()

            while len(test_batch_x) != 0:
                test_prediction = sess.run(prediction, feed_dict={x: test_batch_x})
                for i in range(np.shape(test_batch_y)[0]):
                    if class_names_test[np.argmax(test_batch_y[i])] == class_names_model[np.argmax(test_prediction[i])]:
                        correct += 1
                        if show_test and (epoch + 1) % 10 == 0:
                            cv2.imshow('correct', test_batch_x[i])
                            cv2.waitKey(0)
                    else:
                        wrong += 1
                        if show_test and (epoch + 1) % 10 == 0:
                            cv2.imshow('wrong', test_batch_x[i])
                            cv2.waitKey(0)
                test_batch_x, test_batch_y = data_manager.next_test_batch()
            test_accuracy = 100 * correct / (correct + wrong) if correct + wrong > 0 else 0
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
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
                save_dir = settings.get_save_path()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + 'epoch' + str(epoch+1) + '.ckpt'
                saver.save(sess, save_path)
                settings.update({'model_save_path': save_path, 'input_checkpoint': save_path})
                print('Model saved at ', save_path)
            elif test_accuracy == 0:  # save if meaningful testing is not possible (e.g. no test data available)
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
