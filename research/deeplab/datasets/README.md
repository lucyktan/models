# How to run the code

## Download the dataset
Register on the [Cityscapes dataset website](https://www.cityscapes-dataset.com) to download the dataset

## Clone the CityscapesScripts repo
Use the following command:
`git clone https://github.com/mcordts/cityscapesScripts.git`

## Directory structure
- build_data.py (current working directiory).
- build_cityscapes_data.py
- build_cityscapes_data_dask.py
- build_cityscapes_data_dask_imagesize.py
- build_cityscapes_data_dask_only_delayed.py
- build_cityscapes_data_dask_only_delayed_imagesize.py
- cityscapes
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

## Requirements
+ Versions 1 requires `dask[array,dataframe,delayed]`
+ Versions 2 requires `dask[array,dataframe,delayed]` and [`imagesize`](https://pypi.org/project/imagesize/)
+ Versions 3 requires `dask[delayed]`
+ Versions 4 requires `dask[delayed]` and [`imagesize`](https://pypi.org/project/imagesize/)

## Commands
Run the following command to initialize the ground truth labels:
`python cityscapesscripts/preparation/createTrainIdLabelImgs.py`

Then run the following command to use the original code to convert the dataset to TFRecords:
`python build_cityscapes_data.py --cityscapes_root=cityscapes/ --output_dir=tfrecord/`

To use the Dask versions, replace `build_cityscapes_data.py` with one of the following:
1. `build_cityscapes_data_dask.py`
2. `build_cityscapes_data_dask_imagesize.py`
3. `build_cityscapes_data_dask_only_delayed.py`
4. `build_cityscapes_data_dask_only_delayed_imagesize.py`

## Tests
Run the tests with the following command:
`python build_cityscapes_data_test.py`
