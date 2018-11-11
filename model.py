import os
import numpy as np
import tensorflow as tf
from data import DataManager


def model(x, y, dropout_probability, settings):
    """
    This method builds the classification model
    :param x: The input to be processed by the classifier. Expected to be of size [None, settings.height * settings.width]
    :param y: The label corresponding to input x
    :param dropout_probability: placeholder for dropout probability so that it can be utilized dynamically
    :param settings: The settings determining the construction of the model.
    :return: The output produced by the model as well as the loss.
    """
    print('build model:')
    conv_filters = settings.get_setting_by_name('conv_filters')
    pooling_after_conv = settings.get_setting_by_name('pooling_after_conv')
    kernel_sizes = settings.get_setting_by_name('conv_kernels')
    fc_hidden_units = settings.get_setting_by_name('fc_hidden')
    tensor = tf.reshape(x, shape=[-1, settings.get_setting_by_name('height'), settings.get_setting_by_name('width'), 1])
    temp_width = settings.get_setting_by_name('width')
    temp_height = settings.get_setting_by_name('height')
    training_lock = settings.get_setting_by_name('training_lock')
    temp_filters = 1

    print('input: ' + str(temp_width) + ' x ' + str(temp_height) + ' x  1')

    with tf.variable_scope('cnn'):
        for i in range(len(conv_filters)):
            tensor = tf.layers.conv2d(tensor, conv_filters[i], kernel_sizes[i], padding='SAME', activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                      bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=('cnn' not in training_lock))
            temp_filters = conv_filters[i]
            description = 'conv: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(conv_filters[i])
            if 'cnn' in training_lock:
                description += ' locked!'
            print(description)
            if pooling_after_conv[i]:
                tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
                temp_height = int(temp_height / 2)
                temp_width = int(temp_width / 2)
                print('pool: ' + str(temp_width) + ' x ' + str(temp_height) + ' x ' + str(conv_filters[i]))

    tensor = tf.reshape(tensor, shape=[-1, temp_height * temp_width * temp_filters])
    print('resize: ' + str(temp_width) + '*' + str(temp_height) + '*' + str(temp_filters))

    with tf.variable_scope('dnn'):
        for i in range(len(fc_hidden_units)):
            fc = tf.layers.dense(tensor, fc_hidden_units[i], activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                 bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False), trainable=('dnn' not in training_lock))
            tensor = tf.nn.dropout(fc, dropout_probability)
            description = 'fc: ' + str(fc_hidden_units[i])
            if 'dnn' in training_lock:
                description += ' locked!'
            print(description)

    with tf.variable_scope('out'):
        fc = tf.layers.dense(tensor, settings.get_setting_by_name('num_classes'), activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                             bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        tensor = tf.nn.dropout(fc, dropout_probability)
        print('out: ' + str(settings.get_setting_by_name('num_classes')))
        output = tf.nn.softmax(tensor)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(output, y)))

    return output, loss


def train(settings, n_epochs=401, restore_type='', restore_data=''):
    """
    Train a model.
    While the model keeps improving it will be saved to disk.
    Once a model has been saved it can be retrieved in various ways given through restore_type and restore_data.

    :param settings: the settings providing all the necessary information for building the model and gathering the input data.
    :param n_epochs: the number of epochs for which the model will be trained. One epoch utilizes all training material exactly once.
    :param restore_type: restore model and continue training. Either
                        'auto': for automatic continuation
                        'path': for reading save from path given in restore_data
                        'by_name': for restoring from save from settings.save_path dir with name restore_data
                        'transfer': restores only the convolutional layers and retrains the dense layers, restores from path given in restore_data
    :param restore_data: depending on the restore type, restore data must provide the according information - either the name or path of a saved model
    """
    data_manager = DataManager(settings)

    x = tf.placeholder(tf.float32, [None, settings.get_setting_by_name('width') * settings.get_setting_by_name('height')])
    y = tf.placeholder(tf.float32, [None, settings.get_setting_by_name('num_classes')])
    dropout_probability = tf.placeholder_with_default(1.0, shape=())

    prediction, loss = model(x, y, dropout_probability, settings)

    with tf.name_scope('opt'):
        optimizer = tf.train.AdamOptimizer(learning_rate=settings.get_setting_by_name('learning_rate')).minimize(loss)

    tf.summary.scalar("cost", loss)
    merged_summary_op = tf.summary.merge_all()

    # config = tf.ConfigProto(device_count={'GPU': 0}) # train without gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())
        # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph()) # TODO writer
        if (restore_type == 'auto' or restore_type == 'AUTO') and settings.get_setting_by_name('model_save_path') is not None:
            saver.restore(sess, settings.get_setting_by_name('model_save_path'))
            print('model restored from ', settings.get_setting_by_name('model_save_path'))
        elif restore_type == 'by_name':
            saver.restore(sess, settings.models_path + settings.get_setting_by_name('model_name') + os.sep + 'saves' + os.sep + str(restore_data) + '.ckpt')
            print('model restored from ' + settings.models_path + settings.get_setting_by_name('model_name') + os.sep + 'saves' + os.sep + str(restore_data) + '.ckpt')
        elif restore_type == 'path':
            saver.restore(sess, restore_data)
            print('model restored from ', restore_data)
        elif restore_type == 'transfer':
            training_lock = settings.get_setting_by_name('training_lock')
            if 'cnn' in training_lock:
                cnn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn'))
                cnn_saver.restore(sess, restore_data)
                print('CNN restored from ', restore_data, ' and LOCKED!')
            if 'dnn' in training_lock:
                dnn_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dnn'))
                dnn_saver.restore(sess, restore_data)
                print('DNN restored from ', restore_data, ' and LOCKED!')


        min_cost = -1
        new_best = False
        no_improvement_counter = 0
        last_save_counter = 0
        for epoch in range(n_epochs+1):
            loss_sum = 0
            counter = 0
            correct = 0
            wrong = 0
            batch_x, batch_y = data_manager.next_batch()

            while batch_x != []:
                _prediction, _, c, summary = sess.run([prediction, optimizer, loss, merged_summary_op], feed_dict={x: batch_x, y:batch_y, dropout_probability: 0.8})
                loss_sum += c
                counter += 1
                # writer.add_summary(summary, epoch * n_batches + i) # TODO writer
                for i in range(np.shape(batch_y)[0]):
                    # print(batch_y[i], ' ', _prediction[i])
                    if np.argmax(batch_y[i]) == np.argmax(_prediction[i]):
                        correct += 1
                    else:
                        wrong += 1
                batch_x, batch_y = data_manager.next_batch()
            train_accuracy = 'train accuracy: ' + str(round(100 * correct / (wrong + correct), 2))+'%'
            loss_avg = loss_sum  # / counter

            test_batch_x, test_batch_y = data_manager.next_test_batch()
            wrong = 0
            correct = 0

            while test_batch_x != []:
                _prediction, _cost = sess.run([prediction, loss], feed_dict={x: test_batch_x, y: test_batch_y})
                for i in range(np.shape(test_batch_y)[0]):
                    if list(test_batch_y[i]).index(max(test_batch_y[i])) == list(_prediction[i]).index(max(_prediction[i])):
                        correct += 1
                    else:
                        wrong += 1
                test_batch_x, test_batch_y = data_manager.next_test_batch()
            test_accuracy = 'test accuracy: ' + str(round(100*correct/(wrong+correct), 2))+'%'
            train_info = 'Epoch '+str(epoch + 1)+' / '+str(n_epochs-1)+' cost: '+str(round(loss_avg, 3))

            if min_cost > loss_sum or min_cost == -1:
                min_cost = loss_sum
                no_improvement_counter = 0
                train_info += ' :)'
                new_best = True
            else:
                no_improvement_counter += 1
                if no_improvement_counter > 12:
                    train_info += ' :('
                    new_best = False
                else:
                    train_info += ' :|'

            print(train_info, ' ', test_accuracy, ' ', train_accuracy)
            last_save_counter += 1

            if new_best and last_save_counter > 5:
                save_dir = settings.models_path + settings.get_setting_by_name('model_name') + os.sep + 'saves'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = save_dir + os.sep + 'epoch' + str(epoch) + '.ckpt'
                saver.save(sess, save_path)
                settings.update({'model_save_path': save_path})
                print('model saved at ', save_path)
                last_save_counter = 0

        print('Optimization Finished')
        saver.save(sess, "saves/model-final.ckpt")
        sess.close()
