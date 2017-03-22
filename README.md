# SVHNClassifier

A TensorFlow implementation of [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf) 


## Graph



## Results




## Requirements

* Tensorflow 1.0
* h5py

    ```
    In Ubuntu:
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```

## Setup

1. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

2. extract to data folder, now your folder structure should like below:
    ```
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
    ```


## Usage

1. Take a look at original images with bounding boxes (Optional)

    ```
    Open `draw_bbox.ipynb` in Jupyter
    ```

1. Convert to TFRecords format

    ```
    $ python convert_to_tfrecords.py --data_dir ./data
    ```

1. Test for reading TFRecords files (Optional)

    Open `read_tfrecords_sample.ipynb` in Jupyter
    Open `donkey_sample.ipynb` in Jupyter

1. Train

    ```
    $ python train.py --data_dir ./data --train_logdir ./logs/train
    ```

1. Retrain if you need
    ```
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

1. Try to make an inference (Optional)
    
    Open `inference_sample.ipynb` in Jupyter
    Open `inference_sample2.ipynb` in Jupyter

1. Clean

    ```
    $ rm -rf ./logs
    or
    $ rm -rf ./logs/train2
    or
    $ rm -rf ./logs/eval
    ```
