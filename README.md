# SVHNClassifier

## Environment

Tensorflow 1.0

## Setup

1. Download SVHN [Dataset](http://ufldl.stanford.edu/housenumbers/) format 1, train, test, extra

2. extract to data, so now your structure should like this:
    SVHNClassifier
        - data
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat

1. Install h5py

    ```
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```


## Usage

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ./data
    ```

    > **OPTIONAL**
    > Check `draw_bbox.ipynb` for 
    > Check `read_tfrecords_sample.ipynb` for
    > Check `donkey_sample.ipynb` for

1. Train

    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train
    $ python train.py --data_dir ./data --train_logdir ./logs/train2 --restore_checkpoint ./logs/train/latest.ckpt
    ```

1. Evaluate

    ```
    $ python eval.py --data_dir ./data --checkpoint_dir ./logs/train --eval_logdir ./logs/eval
    ```

1. Visualize

    ```
    $ tensorboard --logdir ./logs
    ```

1. (Optional) Check `inference_sample.ipynb`

1. (Optional) Check `inference_sample2.ipynb`

1. Clean trained or eval

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs/train2
    or
    $ rm -rf ./logs/eval
    ```
