"""Flags for building cityscapes data.

Common to all build_cityscapes_data* files
"""
import tensorflow as tf

tf.app.flags.DEFINE_string('cityscapes_root',
                           './cityscapes',
                           'Cityscapes dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')