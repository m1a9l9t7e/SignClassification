import time
import uff

import cv2
import os
import sys
import tensorflow as tf
import numpy as np

import util
from data import DataManager


def freeze_graph(settings, output_dir=None):
    """
    Given a tensorflow graph, remove all nodes irrelevant to output nodes given in settings and
    convert all variables to constants.
    :param settings: settings specifying:
        - 'input_checkpoint': location of checkpoint from which model will be loaded
        - 'output_node_names': names of output nodes of graph
    :param output_dir: output directory that frozen graph will be written to
    """
    print('Freezing graph..')
    if settings.get_setting_by_name('input_checkpoint') is not None:
        input_checkpoint = settings.get_setting_by_name('input_checkpoint')
    else:
        print('ERROR: model has to first be trained/restored from checkpoint before it can be frozen!')
        print('Aborting.')
        sys.exit(0)

    output_node_names = settings.get_setting_by_name('output_node_name')
    output_dir = settings.get_output_path() if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = input_checkpoint.split(os.sep)[-1].split('.')[0] + 'frozen.pb'
    output_filename = os.path.join(output_dir, name)

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)

        # Use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the useful nodes
        )
        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print(len(output_graph_def.node), ' ops in the final graph')
        print('Frozen graph saved at ', output_filename)
        settings.update({'frozen_model_save_path': os.path.abspath(output_filename)})

    return output_graph_def


def load_graph(frozen_graph_filename):
    print('Loading graph..')
    # get graph definition
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # load graph via graph definition. name='' is important!
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def execute_frozen_model(settings, data_manager):

    graph = load_graph(settings.get_setting_by_name('frozen_model_save_path'))
    output_node_name = settings.get_setting_by_name('output_node_name')
    input_node_name = settings.get_setting_by_name('input_node_name')

    input_node_in_graph = False
    output_node_in_graph = False
    graph_nodes = graph.get_operations()
    print('input node name: ', graph_nodes[0].name)
    print('output node name: ', graph_nodes[-1].name)

    for op in graph.get_operations():
        if op.name == input_node_name:
            input_node_in_graph = True
        if op.name == output_node_name:
            output_node_in_graph = True
    if not (input_node_in_graph and output_node_in_graph):
        print('Input or output node specified in settings is not in graph!')
        print('Abort.')
        sys.exit(0)

    x = graph.get_tensor_by_name(input_node_name+':0')
    y = graph.get_tensor_by_name(output_node_name+':0')

    print('Executing frozen model..')
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything as there are no Variables in this graph, only hardcoded constants
        test_batch_x, test_batch_y = data_manager.next_test_batch()

        while len(test_batch_x) > 0:
            timestamp = time.time()
            predictions = sess.run(y, feed_dict={x: test_batch_x})
            evaluation_time = time.time() - timestamp
            print('Evaluation time of ', evaluation_time, 'seconds, for ', len(test_batch_x), ' images.')

            correct = 0
            wrong = 0
            for i in range(len(test_batch_y)):
                if np.argmax(test_batch_y[i]) == np.argmax(predictions[i]):
                    correct += 1
                    cv2.imshow('correct', test_batch_x[i])
                else:
                    wrong += 1
                    cv2.imshow('wrong', test_batch_x[i])
                cv2.waitKey(0)
            test_batch_x, test_batch_y = data_manager.next_test_batch()
    print('Execution finished.')


def test_frozen_model(settings, data_manager):
    test_batch_x, test_batch_y = data_manager.next_test_batch()
    for i in range(len(test_batch_x)):
        windows = util.sliding_window(image=test_batch_x[i], width=settings.get_setting_by_name('width'), height=settings.get_setting_by_name('height'), stride=10)


def convert_frozen_to_uff(settings, output_dir=None):
    output_dir = settings.get_output_path() if output_dir is None else output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    name = settings.get_setting_by_name('frozen_model_save_path').split(os.sep)[-1].split('.')[0] + 'frozen.uff'
    output_filename = os.path.join(output_dir, name)

    converted = uff.from_tensorflow_frozen_model(frozen_file=settings.get_setting_by_name('frozen_model_save_path'),
                                                 output_nodes=[settings.get_setting_by_name('output_node_name')], preprocessor=None)
    with tf.gfile.GFile(output_filename, "wb") as f:
        f.write(converted)
