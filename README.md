# OrganSegC2F: a coarse-to-fine organ segmentation framework
version 1.10 - Sep 21 2017 - by Lingxi Xie

If you use our codes, please cite our paper accordingly:
  Yuyin Zhou, Lingxi Xie, Wei Shen, Yan Wang, Elliot K. Fishman, Alan L. Yuille,
    "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",
    in International Conference on MICCAI, Quebec City, Quebec, Canada, 2017.

https://arxiv.org/abs/1612.08230

http://lingxixie.com/Projects/OrganSegC2F.html

All the materials released in this library can ONLY be used for RESEARCH purposes.
  The authors preserve the copyright and all legal rights of these codes.


## 1. Introduction


OrganSegC2F is a code package for our paper:
    Yuyin Zhou, Lingxi Xie, Wei Shen, Yan Wang, Elliot Fishman, Alan Yuille,
        "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",
        in International Conference on MICCAI, Quebec City, Quebec, 2017.

OrganSegC2F is a segmentation framework designed for 3D volumes.
    It was originally designed for segmenting abdominal organs in CT scans,
    but we believe that it can also be used for other purposes,
    such as brain tissue segmentation in fMRI-scanned images.

OrganSegC2F is based on the state-of-the-art deep learning techniques.
    This code package is to be used with CAFFE, a deep learning library.
    We make use of the python interface of CAFFE, named pyCAFFE.

It is highly recommended to use one or more modern GPUs for computation.
    Using CPUs will cost at least 50x more time in computation.


## 2. File List

| Folder/File                | Description                                          |
|:-------------------------- |:---------------------------------------------------- |
| `README.txt`               | the README file                                      |
|                            |                                                      |
| **DATA2NPY/**              | codes to transfer the NIH dataset into NPY format    |
| `dicom2npy.py`             | transferring image data (DICOM) into NPY format      |
| `nii2npy.py`               | transferring label data (NII) into NPY format        |
|                            |                                                      |
| **DiceLossLayer/**         | CPU implementation of the Dice loss layer            |
| `dice_loss_layer.hpp`      | the header file                                      |
| `dice_loss_layer.cpp`      | the CPU implementation                               |
|                            |                                                      |
| **OrganSegC2F/**           | primary codes of OrganSegC2F                         |
| `coarse2fine_testing.py`   | the coarse-to-fine testing process                   |
| `coarse_fusion.py`         | the coarse-scaled fusion process                     |
| `coarse_surgery.py`        | the surgery function for coarse-scaled training      |
| `coarse_testing.py`        | the coarse-scaled testing process                    |
| `coarse_training.py`       | the coarse-scaled training process                   |
| `DataC.py`                 | the data layer in the coarse-scaled training         |
| `DataF.py`                 | the data layer in the fine-scaled training           |
| `fine_surgery.py`          | the surgery function for fine-scaled training        |
| `fine_training.py`         | the fine-scaled training process                     |
| `init.py`                  | the initialization functions                         |
| `oracle_fusion.py`         | the fusion process with oracle information           |
| `oracle_testing.py`        | the testing process with oracle information          |
| `run.sh`                   | the main program to be called in bash shell          |
| `utils.py`                 | the common functions                                 |
|                            |                                                      |
| **OrganSegC2F/prototxts**  | primary codes of OrganSegC2F                         |
| `deploy_1.prototxt`        | the prototxt file for 1-slice testing                |
| `deploy_3.prototxt`        | the prototxt file for 3-slice testing                |
| `training_C1.prototxt`     | the prototxt file for 1-slice coarse-scaled training |
| `training_C3.prototxt`     | the prototxt file for 3-slice coarse-scaled training |
| `training_F1.prototxt`     | the prototxt file for 1-slice fine-scaled training   |
| `training_F3.prototxt`     | the prototxt file for 3-slice fine-scaled training   |


## 3. Installation


#### 3.1 Prerequisites

###### 3.1.1 Please make sure that your computer is equipped with modern GPUs that support CUDA.
    Without them, you will need 50x more time in both training and testing stages.

###### 3.1.2 Please also make sure that python (we are using 2.7) is installed.


#### 3.2 CAFFE and pyCAFFE

###### 3.2.1 Download a CAFFE library from http://caffe.berkeleyvision.org/ .
    Suppose your CAFFE root directory is $CAFFE_PATH.

###### 3.2.2 Place the files of Dice loss layer at the correct position.
    dice_loss_layer.hpp -> $CAFFE_PATH/include/caffe/layers/
    dice_loss_layer.cpp -> $CAFFE_PATH/src/caffe/layers/

###### 3.2.3 Make CAFFE and pyCAFFE.


## 4. Usage

Please follow these steps to reproduce our results on the NIH pancreas segmentation dataset.

NOTE: Here we only provide basic steps to run our codes on the NIH dataset.
    For more detailed analysis and empirical guidelines for parameter setting
    (this is very important especially when you are using our codes on other datasets),
    please refer to our technical report (check our webpage for updates).


#### 4.1 Data preparation

###### 4.1.1 Download NIH data from https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT .
    You should be able to download image and label data individually.
    Suppose your data directory is $RAW_PATH:
        The image data are organized as $RAW_PATH/DOI/PANCREAS_00XX/A_LONG_CODE/A_LONG_CODE/ .
        The label data are organized as $RAW_PATH/TCIA_pancreas_labels-TIMESTAMP/label00XX.nii.gz .

###### 4.1.2 Use our codes to transfer these data into NPY format.
    Put dicom2npy.py under $RAW_PATH, and run: python dicom2npy.py .
        The transferred data should be put under $RAW_PATH/images/
    Put nii2npy.py under $RAW_PATH, and run: python nii2npy.py .
        The transferred data should be put under $RAW_PATH/labels/

###### 4.1.3 Suppose your directory to store experimental data is $DATA_PATH:
    Put $CAFFE_PATH under $DATA_PATH/libs/
    Put images/ under $DATA_PATH/
    Put labels/ under $DATA_PATH/

    NOTE: If you use other path(s), please modify the variable(s) in run.sh accordingly.


#### 4.2 Initialization (requires: 4.1)

###### 4.2.1 Check run.sh and set $DATA_PATH accordingly.

###### 4.2.2 Set $ENABLE_INITIALIZATION=1 and run this script.
    Several folders will be created under $DATA_PATH:
        $DATA_PATH/images_X|Y|Z: the sliced image data (data are sliced for faster I/O).
        $DATA_PATH/labels_X|Y|Z: the sliced label data (data are sliced for faster I/O).
        $DATA_PATH/lists: used for storing training, testing and slice lists.
        $DATA_PATH/logs: used for storing log files during the training process.
        $DATA_PATH/models: used for storing models (snapshots) during the training process.
        $DATA_PATH/prototxts: used for storing prototxts (called by training and testing nets).
        $DATA_PATH/results: used for storing testing results (volumes and text results).
    According to the I/O speed of your hard drive, the time cost may vary.
        For a typical HDD, around 20 seconds are required for a 512x512x300 volume.
    This process needs to be executed only once.

    NOTE: if you are using another dataset which contains multiple targets,
        you can modify the variables "ORGAN_NUMBER" and "ORGAN_ID" in run.sh,
        as well as the "is_organ" function in utils.py to define your mapping function flexibly.


![alt text](https://github.com/198808xc/OrganSegC2F/blob/master/icon.png)
**LAZY MODE!**
![alt text](https://github.com/198808xc/OrganSegC2F/blob/master/icon.png)

You can run all the following modules with **one** execution!
a) Enable everything (except initialization) in the beginning part.
b) Set all the "PLANE" variables as "A" (4 in total) in the following part.
c) Run this manuscript!


#### 4.3 Coarse-scaled training (requires: 4.2)

###### 4.3.1 Check run.sh and set $COARSE_TRAINING_PLANE and $COARSE_TRAINING_GPU.
    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set COARSE_TRAINING_PLANE=A, so that three planes are trained orderly in one GPU.

###### 4.3.2 Set $ENABLE_COARSE_TRAINING=1 and run this script.
    The following folders/files will be created:
        Under $DATA_PATH/logs/, a log file named by training information.
        Under $DATA_PATH/models/snapshots/, a folder named by training information.
            Snapshots and solver-states will be stored in this folder.
            The log file will also be copied into this folder after the entire training process.
    On the axial view (training image size is 512x512, small input images make training faster),
        each 20 iterations cost ~8s on a Titan-X Maxwell GPU, or ~5s on a Titan-X Pascal GPU.
        As described in the paper, we need ~80K iterations, which take less than 10 GPU-hours.
    After the training process, the log file will be copied to the snapshot directory.


#### 4.4 Coarse-scaled testing (requires: 4.3)

###### 4.4.1 Check run.sh and set $COARSE_TESTING_PLANE and $COARSE_TESTING_GPU.
    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set COARSE_TESTING_PLANE=A, so that three planes are tested orderly in one GPU.

###### 4.4.2 Set $ENABLE_COARSE_TESTING=1 and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by training information.
    Testing each volume costs ~20 seconds on a Titan-X Maxwell GPU, or ~13s on a Titan-X Pascal GPU.


#### 4.5 Coarse-scaled fusion (requires: 4.4)

###### 4.5.1 Fusion is perfomed on CPU and all X|Y|Z planes are combined and executed once.

###### 4.5.2 Set $ENABLE_COARSE_FUSION=1 and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by fusion information.
    The main cost in fusion includes I/O and post-processing (removing non-maximum components).
        In our future release, we will implement post-processing in C for acceleration.


#### 4.6 Fine-scaled training (requires: 4.2)

###### 4.6.1 Check run.sh and set $FINE_TRAINING_PLANE and $FINE_TRAINING_GPU.
    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set FINE_TRAINING_PLANE=A, so that three planes are trained orderly in one GPU.

###### 4.6.2 Set $ENABLE_FINE_TRAINING=1 and run this script.
    The following folders/files will be created:
        Under $DATA_PATH/logs/, a log file named by training information.
        Under $DATA_PATH/models/snapshots/, a folder named by training information.
            Snapshots and solver-states will be stored in this folder.
            The log file will also be copied into this folder after the entire training process.
    On the axial view (training image size is ~150x150, small input images make training faster),
        each 20 iterations cost ~5s on a Titan-X Maxwell GPU, or ~3s on a Titan-X Pascal GPU.
        As described in the paper, we need 60K iterations, which take less than 5 GPU-hours.
    After the training process, the log file will be copied to the snapshot directory.


#### 4.7 Oracle testing (optional) (requires: 4.6)

NOTE: Without this step, you can also run the coarse-to-fine testing process.
    This stage is still recommended, so that you can check the quality of the fine-scaled models.

###### 4.7.1 Check run.sh and set $ORACLE_TESTING_PLANE and $ORACLE_TESTING_GPU.
    You need to run X|Y|Z planes individually, so you can use 3 GPUs in parallel.
    You can also set ORACLE_TESTING_PLANE=A, so that three planes are tested orderly in one GPU.

###### 4.7.2 Set $ENABLE_ORACLE_TESTING=1 and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by training information.
    Testing each volume costs ~5 seconds on a Titan-X Maxwell GPU, or ~3s on a Titan-X Pascal GPU.


#### 4.8 Oracle fusion (optional) (requires: 4.7)

NOTE: Without this step, you can also run the coarse-to-fine testing process.
    This stage is still recommended, so that you can check the quality of the fine-scaled models.

###### 4.8.1 Fusion is perfomed on CPU and all X|Y|Z planes are combined and executed once.

###### 4.8.2 Set $ENABLE_ORACLE_FUSION=1 and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by fusion information.
    The main cost in fusion includes I/O and post-processing (removing non-maximum components).
        In our future release, we will implement post-processing in C for acceleration.


#### 4.9 Coarse-to-fine testing (requires: 4.4 & 4.6)

###### 4.9.1 Check run.sh and set $COARSE2FINE_TESTING_GPU.
    Fusion is performed on CPU and all X|Y|Z planes are combined.
    Currently X|Y|Z testing processes are executed with one GPU, but it is not time-comsuming.

###### 4.9.2 Set $ENABLE_COARSE2FINE_TESTING=1 and run this script.
    The following folder will be created:
        Under $DATA_PATH/results/, a folder named by coarse-to-fine information (very long).
    This function calls both fine-scaled testing and fusion codes, so both GPU and CPU are used.
        In our future release, we will implement post-processing in C for acceleration.

NOTE: currently we set the maximal rounds of iteration to be 10 in order to observe the convergence.
    Most often, it reaches an inter-DSC of >95% after 2-3 iterations.
    If you hope to save time, you can slight modify the codes in coarse2fine_testing.py.
    Each iteration takes ~20 seconds on a Titan-X Maxwell GPU, or ~15s on a Titan-X Pascal GPU.
    If you set the threshold to be 95%, this stage will be done within 1 minute (in average).


Congratulations! You have finished the entire process. Check your results now!


## 5. Versions

The current version is v1.10.

To access old versions, please visit our project page: http://lingxixie.com/Projects/OrganSegC2F.html.

You can also view CHANGE_LOG.txt for the history of versions.


## 6. Contact Information

If you encounter any problems in using these codes, please contact Lingxi Xie (198808xc@gmail.com).

Thanks for your interest! Have fun!
