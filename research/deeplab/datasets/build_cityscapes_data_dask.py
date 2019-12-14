# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts Cityscapes data to TFRecord file format with Example protos.

The Cityscapes dataset is expected to have the following directory structure:

  + cityscapes
     - build_cityscapes_data.py (current working directiory).
     - build_data.py
     + cityscapesscripts
       + annotation
       + evaluation
       + helpers
       + preparation
       + viewer
     + gtFine
       + train
       + val
       + test
     + leftImg8bit
       + train
       + val
       + test
     + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

Note that before running this script, the users should (1) register the
Cityscapes dataset website at https://www.cityscapes-dataset.com to
download the dataset, and (2) run the script provided by Cityscapes
`preparation/createTrainIdLabelImgs.py` to generate the training groundtruth.

Also note that the tensorflow model will be trained with `TrainId' instead
of `EvalId' used on the evaluation server. Thus, the users need to convert
the predicted labels to `EvalId` for evaluation on the server. See the
vis.py for more details.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import math
import os.path
import re
import sys
import build_cityscapes_data_flags
import build_data
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from six.moves import range
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.adopt_module_key_flags(build_cityscapes_data_flags)


_NUM_SHARDS = 10

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.

  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')

  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if data == 'label' and dataset_split == 'test':
    return None
  pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
  search_files = os.path.join(
      FLAGS.cityscapes_root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
  filenames = glob.glob(search_files)
  return sorted(filenames)


def _get_image_dimensions(image_data, reader):
  """Gets the dimensions of the image.

  Args:
    image_data: pandas Series with raw bytes of the image file.
    reader: build_data.ImageReader for parsing the dimensions of the image.

  Returns:
    A pandas DataFrame with the height and width of the image.
  """
  height, width = reader.read_image_dims(image_data.values[0])
  data = {'height': [height], 'width': [width]}
  return pd.DataFrame(data, index=image_data.index)


@dask.delayed(pure=True)
def _get_image_data(path):
  """Gets the raw bytes of the given image path.

  Note that it returns a numpy array so it can be read into dask.

  Args:
    path: Path to the image file.

  Returns:
    A numpy array with a single element as the raw bytes of the image.
  """
  return np.array(tf.gfile.FastGFile(path, 'rb').read())


def _get_bytes_series(filenames):
  """Gets the raw bytes of the given image paths.

  Args:
    filenames: List of paths to image files.

  Returns:
    A dask Series with the raw bytes of each image.
  """
  # Note that using np.bytes_ as the dtype results in it becoming S1, which
  # would truncate it to only the first byte of each file.
  byte_data_arrays = [
      da.from_delayed(_get_image_data(path), dtype=object, shape=())
      for path in filenames
  ]
  byte_data = da.stack(byte_data_arrays, axis=0)
  return dd.from_dask_array(byte_data, columns='data')


def _create_tf_record(record):
  """Creates a serialized TFRecord.

  Args:
    record: Pandas Series of data needed to construct a TFRecord.

  Returns:
    Bytes representing the serialized TFRecord.
  """
  return build_data.image_seg_to_tfexample(
      record['img_data'], record['filename'], record['image_height'],
      record['image_width'], record['seg_data']).SerializeToString()


def _convert_to_record(df):
  """Gets the serialized TFRecord.

  Args:
    df: Pandas DataFrame to create a TFRecord from.

  Returns:
    A pandas Series containing the serialized TFRecord.
  """
  return df.apply(_create_tf_record, axis=1)


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, val).

  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)

  num_images = len(image_files)
  num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

  image_reader = build_data.ImageReader('png', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  image_file_array = np.char.array(
      list(map(lambda f: os.path.basename(f), image_files)))
  pattern = '%s.%s' % (_POSTFIX_MAP['image'], _DATA_FORMAT_MAP['image'])
  bad_images = image_file_array[~image_file_array.endswith(pattern)]
  if bad_images.size:
    raise RuntimeError('Invalid image filenames: %s' % bad_images)

  # Remove the ending pattern to match the filename used in the original code.
  ddf = dd.from_array(image_file_array).str[:-len(pattern)].to_frame(
      name='filename')

  ddf['img_data'] = _get_bytes_series(image_files)
  ddf['seg_data'] = _get_bytes_series(label_files)

  ddf[['image_height', 'image_width']] = ddf['img_data'].map_partitions(
      _get_image_dimensions,
      reader=image_reader,
      meta={
          'height': np.int64,
          'width': np.int64
      })

  ddf[['label_height', 'label_width']] = ddf['seg_data'].map_partitions(
      _get_image_dimensions,
      reader=label_reader,
      meta={
          'height': np.int64,
          'width': np.int64
      })

  records = ddf.map_partitions(_convert_to_record, meta=('record', object))

  for shard_id in range(_NUM_SHARDS):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      cur_records = records.loc[start_idx:end_idx - 1].compute()
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        tfrecord_writer.write(cur_records[i])
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
