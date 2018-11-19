import os
import sys
import tensorflow as tf


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
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        # Serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print(len(output_graph_def.node), ' ops in the final graph')
        print('Frozen graph saved at ', output_filename)

    return output_graph_def
