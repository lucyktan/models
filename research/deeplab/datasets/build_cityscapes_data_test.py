import itertools
import glob
import os
import shutil
import tempfile

import tensorflow as tf

import build_cityscapes_data
import build_cityscapes_data_dask
import build_cityscapes_data_dask_imagesize

FLAGS = tf.app.flags.FLAGS

# Maps each feature name to its type. Based on build_data.image_seg_to_tfexample.
_FEATURES = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/channels': tf.io.FixedLenFeature([], tf.int64),
    'image/segmentation/class/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/segmentation/class/format': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
  """Parse a tf.Example proto as a dictionary of features.

  Args:
    example_proto: The tf.Example proto containing all the features.

  Returns:
    A dictionary mapping each feature name to a tensor with its value.
  """
  return tf.io.parse_single_example(example_proto, _FEATURES)


def _copy_images(temp_dir):
  """Copy images to a temporary directory.

  Args:
    temp_dir: The root temporary directory to store the images.
  """
  images_dir = os.path.join(temp_dir, 'cityscapes/leftImg8bit/train/erfurt')
  labels_dir = os.path.join(temp_dir, 'cityscapes/gtFine/train/erfurt')
  os.makedirs(images_dir)
  os.makedirs(labels_dir)
  for img in os.listdir('./testdata/leftImg8bit/'):
    shutil.copy2(os.path.join('./testdata/leftImg8bit', img), images_dir)
  for img in os.listdir('./testdata/gtFine/'):
    shutil.copy2(os.path.join('./testdata/gtFine', img), labels_dir)


class BuildCityscapesDataTest(tf.test.TestCase):
  """Tests for converting Cityscapes images to TFRecords"""

  # Allow subclasses to use different modules for conversion.
  module = build_cityscapes_data

  def assertRecordEqual(self, expected, actual):
    """Asserts that two TFRecords are equivalent.

    Args:
      expected: The expected TFRecord.
      actual: The output TFRecord.

    Raises:
      AssertionError: If either `expected` or `actual` is None or any of their
        features do not match.
    """
    self.assertIsNotNone(expected, 'More results than expected')
    self.assertIsNotNone(actual, 'Fewer results than expected')
    for feature in _FEATURES:
      self.assertEqual(expected[feature].numpy(), actual[feature].numpy())

  def assertAllRecordsEqual(self, expected_records_path, actual_records_path):
    """Asserts that two TFRecordDatasets are equivalent.

    Args:
      expected_records_path: The path to the expected TFRecordDataset.
      actual_records_path: The path to the output TFRecordDataset.
    """
    expected_records = tf.data.TFRecordDataset(glob.glob(expected_records_path))
    actual_records = tf.data.TFRecordDataset(glob.glob(actual_records_path))
    parsed_expected_records = expected_records.map(_parse_image_function)
    parsed_actual_records = actual_records.map(_parse_image_function)
    for expected_record, actual_record in itertools.zip_longest(
        parsed_expected_records, parsed_actual_records):
      self.assertRecordEqual(expected_record, actual_record)

  def testConvertDataset(self):
    with tempfile.TemporaryDirectory() as tmp:
      _copy_images(tmp)
      FLAGS.cityscapes_root = os.path.join(tmp, 'cityscapes')
      FLAGS.output_dir = os.path.join(tmp, 'tfrecord')
      os.makedirs(FLAGS.output_dir)
      self.module._convert_dataset('train')
      self.assertAllRecordsEqual('./testdata/tfrecord/*.tfrecord',
                                 os.path.join(tmp, 'tfrecord/*.tfrecord'))


class BuildCityscapesDataDaskTest(BuildCityscapesDataTest):
  """Tests dask method of conversion"""
  module = build_cityscapes_data_dask


class BuildCityscapesDataDaskImageSizeTest(BuildCityscapesDataTest):
  """Tests dask + imagesize method of conversion"""
  module = build_cityscapes_data_dask_imagesize


if __name__ == '__main__':
  # Needs to be set at beginning of program when using TF 1 to ensure the
  # tensors have values.
  tf.enable_eager_execution()
  tf.test.main()
