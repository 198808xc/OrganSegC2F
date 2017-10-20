####################################################################################################
# OrganSegC2F: a coarse-to-fine organ segmentation framework                                       #
#                                                                                                  #
# If you use our codes, please cite our paper accordingly:                                         #
#     Yuyin Zhou, Lingxi Xie, Wei Shen, Yan Wang, Elliot K. Fishman, Alan L. Yuille,               #
#         "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans",                   #
#         in International Conference on MICCAI, Quebec City, Quebec, Canada, 2017.                #
#                                                                                                  #
# NOTE: this program can be used on multi-organ segmentation datasets!                             #
#     Please check the ORGAN_NUMBER and ORGAN_ID variables.                                        #
####################################################################################################


####################################################################################################
# turn on these swithes to execute each module
ENABLE_INITIALIZATION=0
ENABLE_COARSE_TRAINING=0
ENABLE_COARSE_TESTING=0
ENABLE_COARSE_FUSION=0
ENABLE_FINE_TRAINING=0
ENABLE_ORACLE_TESTING=0
ENABLE_ORACLE_FUSION=0
ENABLE_COARSE2FINE_TESTING=0
# coarse_training settings: X|Y|Z
COARSE_TRAINING_ORGAN_ID=1
COARSE_TRAINING_PLANE=A
COARSE_TRAINING_GPU=0
# coarse_testing settings: X|Y|Z, before this, coarse-scaled models shall be ready
COARSE_TESTING_ORGAN_ID=1
COARSE_TESTING_PLANE=A
COARSE_TESTING_GPU=0
# coarse_fusion settings: before this, coarse-scaled results on 3 views shall be ready
COARSE_FUSION_ORGAN_ID=1
# fine_training settings: X|Y|Z
FINE_TRAINING_ORGAN_ID=1
FINE_TRAINING_PLANE=A
FINE_TRAINING_GPU=0
# oracle_testing settings: X|Y|Z, before this, fine-scaled models shall be ready
ORACLE_TESTING_ORGAN_ID=1
ORACLE_TESTING_PLANE=A
ORACLE_TESTING_GPU=0
# oracle_fusion settings: before this, fine-scaled results on 3 views shall be ready
ORACLE_FUSION_ORGAN_ID=1
# fine_testing settings: before this, both coarse-scaled and fine-scaled models shall be ready
COARSE2FINE_TESTING_ORGAN_ID=1
COARSE2FINE_TESTING_GPU=0


####################################################################################################
# defining the root path which stores image and label data
DATA_PATH='/media/Med_4T2/data2/'

####################################################################################################
# export PYTHONPATH (related to your path to CAFFE)
export PYTHONPATH=${DATA_PATH}libs/caffe-master/python:$PYTHONPATH

####################################################################################################
# data initialization: only needs to be run once
# variables
ORGAN_NUMBER=1
FOLDS=4
LOW_RANGE=-100
HIGH_RANGE=240
# init.py : data_path, organ_number, folds, low_range, high_range
if [ "$ENABLE_INITIALIZATION" = "1" ]
then
    python init.py \
        $DATA_PATH $ORGAN_NUMBER $FOLDS $LOW_RANGE $HIGH_RANGE
fi

####################################################################################################
# defining the current fold: this parameter will be used in all the following modules
CURRENT_FOLD=0

####################################################################################################
# the coarse-scaled training process
# variables
COARSE_SLICE_THRESHOLD=0.98
COARSE_SLICE_THICKNESS=3
COARSE_LEARNING_RATE=1e-5
COARSE_GAMMA=0.5
COARSE_TRAINING_STARTING_ITERATIONS=0
COARSE_TRAINING_STEP=5000
COARSE_TRAINING_MAX_ITERATIONS=80000
COARSE_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
# coarse_training.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness,
#     organ_ID, plane, GPU_ID,
#     learning_rate, gamma, starting_iterations, step, max_iterations, timestamp
if [ "$ENABLE_COARSE_TRAINING" = "1" ]
then
    if [ "$COARSE_TRAINING_PLANE" = "X" ] || [ "$COARSE_TRAINING_PLANE" = "A" ]
    then
        COARSE_MODELNAME=XC${COARSE_SLICE_THICKNESS}_${COARSE_TRAINING_ORGAN_ID}
        COARSE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${COARSE_MODELNAME}_${COARSE_TIMESTAMP}.txt
        python coarse_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $COARSE_SLICE_THRESHOLD $COARSE_SLICE_THICKNESS \
            $COARSE_TRAINING_ORGAN_ID X $COARSE_TRAINING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TRAINING_STARTING_ITERATIONS $COARSE_TRAINING_STEP \
            $COARSE_TRAINING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP 2>&1 | tee $COARSE_LOG
    fi
    if [ "$COARSE_TRAINING_PLANE" = "Y" ] || [ "$COARSE_TRAINING_PLANE" = "A" ]
    then
        COARSE_MODELNAME=YC${COARSE_SLICE_THICKNESS}_${COARSE_TRAINING_ORGAN_ID}
        COARSE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${COARSE_MODELNAME}_${COARSE_TIMESTAMP}.txt
        python coarse_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $COARSE_SLICE_THRESHOLD $COARSE_SLICE_THICKNESS \
            $COARSE_TRAINING_ORGAN_ID Y $COARSE_TRAINING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TRAINING_STARTING_ITERATIONS $COARSE_TRAINING_STEP \
            $COARSE_TRAINING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP 2>&1 | tee $COARSE_LOG
    fi
    if [ "$COARSE_TRAINING_PLANE" = "Z" ] || [ "$COARSE_TRAINING_PLANE" = "A" ]
    then
        COARSE_MODELNAME=ZC${COARSE_SLICE_THICKNESS}_${COARSE_TRAINING_ORGAN_ID}
        COARSE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${COARSE_MODELNAME}_${COARSE_TIMESTAMP}.txt
        python coarse_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $COARSE_SLICE_THRESHOLD $COARSE_SLICE_THICKNESS \
            $COARSE_TRAINING_ORGAN_ID Z $COARSE_TRAINING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TRAINING_STARTING_ITERATIONS $COARSE_TRAINING_STEP \
            $COARSE_TRAINING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP 2>&1 | tee $COARSE_LOG
    fi
fi

####################################################################################################
# the coarse-scaled testing process
# variables
COARSE_TESTING_STARTING_ITERATIONS=80000
COARSE_TESTING_STEP=5000
COARSE_TESTING_MAX_ITERATIONS=80000
COARSE_TIMESTAMP=_
# coarse_testing.py : data_path, current_fold,
#     organ_number, low_range, high_range, organ_ID, plane,
#     GPU_ID, learning_rate, gamma, starting_iterations, step, max_iterations, timestamp (optional)
if [ "$ENABLE_COARSE_TESTING" = "1" ]
then
    if [ "$COARSE_TESTING_PLANE" = "X" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE $COARSE_SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID X $COARSE_TESTING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP
    fi
    if [ "$COARSE_TESTING_PLANE" = "Y" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE $COARSE_SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID Y $COARSE_TESTING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP
    fi
    if [ "$COARSE_TESTING_PLANE" = "Z" ] || [ "$COARSE_TESTING_PLANE" = "A" ]
    then
        python coarse_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE $COARSE_SLICE_THICKNESS \
            $COARSE_TESTING_ORGAN_ID Z $COARSE_TESTING_GPU \
            $COARSE_LEARNING_RATE $COARSE_GAMMA \
            $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP \
            $COARSE_TESTING_MAX_ITERATIONS \
            $COARSE_TIMESTAMP
    fi
fi

####################################################################################################
# the coarse-scaled fusion process
# variables
COARSE_FUSION_STARTING_ITERATIONS=80000
COARSE_FUSION_STEP=5000
COARSE_FUSION_MAX_ITERATIONS=80000
COARSE_THRESHOLD=0.5
COARSE_TIMESTAMP_X=_
COARSE_TIMESTAMP_Y=_
COARSE_TIMESTAMP_Z=_
# coarse_fusion.py : data_path, current_fold, organ_number, slice_thickness,
#     organ_ID, learning_rate, gamma,
#     starting_iterations, step, max_iterations,
#     threshold, timestamp_X (optional), timestamp_Y (optional), timestamp_Z (optional)
if [ "$ENABLE_COARSE_FUSION" = "1" ]
then
    python coarse_fusion.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $COARSE_SLICE_THICKNESS \
        $COARSE_FUSION_ORGAN_ID $COARSE_LEARNING_RATE $COARSE_GAMMA \
        $COARSE_FUSION_STARTING_ITERATIONS $COARSE_FUSION_STEP $COARSE_FUSION_MAX_ITERATIONS \
        $COARSE_THRESHOLD \
        $COARSE_TIMESTAMP_X $COARSE_TIMESTAMP_Y $COARSE_TIMESTAMP_Z
fi

####################################################################################################
# the fine-scaled training process
# variables
FINE_SLICE_THRESHOLD=1
FINE_SLICE_THICKNESS=3
FINE_TRAINING_MARGIN=20
FINE_RANDOM_PROB=0.5
FINE_SAMPLE_BATCH=1
FINE_LEARNING_RATE=1e-4
FINE_GAMMA=0.5
FINE_TRAINING_STARTING_ITERATIONS=0
FINE_TRAINING_STEP=5000
FINE_TRAINING_MAX_ITERATIONS=60000
FINE_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
FINE_MODELNAME=${FINE_TRAINING_PLANE}F${FINE_SLICE_THICKNESS}_${FINE_TRAINING_ORGAN_ID}
FINE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${FINE_MODELNAME}_${FINE_TIMESTAMP}.txt
# fine_training.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_threshold, slice_thickness,
#     training_margin, random_prob, sample_batch,
#     organ_ID, plane, GPU_ID,
#     learning_rate, gamma, starting_iterations, step, max_iterations, timestamp
if [ "$ENABLE_FINE_TRAINING" = "1" ]
then
    if [ "$FINE_TRAINING_PLANE" = "X" ] || [ "$FINE_TRAINING_PLANE" = "A" ]
    then
        FINE_MODELNAME=XF${FINE_SLICE_THICKNESS}_${FINE_TRAINING_ORGAN_ID}
        FINE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${FINE_MODELNAME}_${FINE_TIMESTAMP}.txt
        python fine_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THRESHOLD $FINE_SLICE_THICKNESS \
            $FINE_TRAINING_MARGIN $FINE_RANDOM_PROB $FINE_SAMPLE_BATCH \
            $FINE_TRAINING_ORGAN_ID X $FINE_TRAINING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $FINE_TRAINING_STARTING_ITERATIONS $FINE_TRAINING_STEP $FINE_TRAINING_MAX_ITERATIONS \
            $FINE_TIMESTAMP 2>&1 | tee $FINE_LOG
    fi
    if [ "$FINE_TRAINING_PLANE" = "Y" ] || [ "$FINE_TRAINING_PLANE" = "A" ]
    then
        FINE_MODELNAME=YF${FINE_SLICE_THICKNESS}_${FINE_TRAINING_ORGAN_ID}
        FINE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${FINE_MODELNAME}_${FINE_TIMESTAMP}.txt
        python fine_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THRESHOLD $FINE_SLICE_THICKNESS \
            $FINE_TRAINING_MARGIN $FINE_RANDOM_PROB $FINE_SAMPLE_BATCH \
            $FINE_TRAINING_ORGAN_ID Y $FINE_TRAINING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $FINE_TRAINING_STARTING_ITERATIONS $FINE_TRAINING_STEP $FINE_TRAINING_MAX_ITERATIONS \
            $FINE_TIMESTAMP 2>&1 | tee $FINE_LOG
    fi
    if [ "$FINE_TRAINING_PLANE" = "Z" ] || [ "$FINE_TRAINING_PLANE" = "A" ]
    then
        FINE_MODELNAME=ZF${FINE_SLICE_THICKNESS}_${FINE_TRAINING_ORGAN_ID}
        FINE_LOG=${DATA_PATH}logs/FD${CURRENT_FOLD}:${FINE_MODELNAME}_${FINE_TIMESTAMP}.txt
        python fine_training.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THRESHOLD $FINE_SLICE_THICKNESS \
            $FINE_TRAINING_MARGIN $FINE_RANDOM_PROB $FINE_SAMPLE_BATCH \
            $FINE_TRAINING_ORGAN_ID Z $FINE_TRAINING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $FINE_TRAINING_STARTING_ITERATIONS $FINE_TRAINING_STEP $FINE_TRAINING_MAX_ITERATIONS \
            $FINE_TIMESTAMP 2>&1 | tee $FINE_LOG
    fi
fi

####################################################################################################
# the fine-scaled testing process with oracle information
# variables
ORACLE_TESTING_MARGIN=$FINE_TRAINING_MARGIN
ORACLE_TESTING_STARTING_ITERATIONS=60000
ORACLE_TESTING_STEP=5000
ORACLE_TESTING_MAX_ITERATIONS=60000
ORACLE_TIMESTAMP=_
# oracle_testing.py : data_path, current_fold, organ_number, low_range, high_range,
#     slice_thickness, training_margin, testing_margin, organ_ID, plane, GPU_ID,
#     learning_rate, gamma, starting_iterations, step, max_iterations, timestamp (optional)
if [ "$ENABLE_ORACLE_TESTING" = "1" ]
then
    if [ "$ORACLE_TESTING_PLANE" = "X" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THICKNESS $FINE_TRAINING_MARGIN $ORACLE_TESTING_MARGIN \
            $ORACLE_TESTING_ORGAN_ID X $ORACLE_TESTING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP
    fi
    if [ "$ORACLE_TESTING_PLANE" = "Y" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THICKNESS $FINE_TRAINING_MARGIN $ORACLE_TESTING_MARGIN \
            $ORACLE_TESTING_ORGAN_ID Y $ORACLE_TESTING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP
    fi
    if [ "$ORACLE_TESTING_PLANE" = "Z" ] || [ "$ORACLE_TESTING_PLANE" = "A" ]
    then
        python oracle_testing.py \
            $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
            $FINE_SLICE_THICKNESS $FINE_TRAINING_MARGIN $ORACLE_TESTING_MARGIN \
            $ORACLE_TESTING_ORGAN_ID Z $ORACLE_TESTING_GPU \
            $FINE_LEARNING_RATE $FINE_GAMMA \
            $ORACLE_TESTING_STARTING_ITERATIONS $ORACLE_TESTING_STEP \
            $ORACLE_TESTING_MAX_ITERATIONS \
            $ORACLE_TIMESTAMP
    fi
fi

####################################################################################################
# the fine-scaled fusion process with oracle information
# variables
ORACLE_FUSION_STARTING_ITERATIONS=60000
ORACLE_FUSION_STEP=5000
ORACLE_FUSION_MAX_ITERATIONS=60000
ORACLE_THRESHOLD=0.5
ORACLE_TIMESTAMP_X=_
ORACLE_TIMESTAMP_Y=_
ORACLE_TIMESTAMP_Z=_
# orcale_fusion.py : data_path, current_fold, organ_number,
#     slice_thickness, training_margin, testing_margin, organ_ID, learning_rate, gamma,
#     starting_iterations, step, max_iterations,
#     threshold, timestamp_X (optional), timestamp_Y (optional), timestamp_Z (optional)
if [ "$ENABLE_ORACLE_FUSION" = "1" ]
then
    python oracle_fusion.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER \
        $FINE_SLICE_THICKNESS $FINE_TRAINING_MARGIN $ORACLE_TESTING_MARGIN \
        $ORACLE_FUSION_ORGAN_ID $FINE_LEARNING_RATE $FINE_GAMMA \
        $ORACLE_FUSION_STARTING_ITERATIONS $ORACLE_FUSION_STEP $ORACLE_FUSION_MAX_ITERATIONS \
        $ORACLE_THRESHOLD \
        $ORACLE_TIMESTAMP_X $ORACLE_TIMESTAMP_Y $ORACLE_TIMESTAMP_Z
fi

####################################################################################################
# the coarse-to-fine testing process
# variables
COARSE_TESTING_STARTING_ITERATIONS=80000
COARSE_TESTING_STEP=5000
COARSE_TESTING_MAX_ITERATIONS=80000
COARSE_THRESHOLD=0.5
COARSE_TIMESTAMP_X=_
COARSE_TIMESTAMP_Y=_
COARSE_TIMESTAMP_Z=_
COARSE_FUSION_CODE=F2
FINE_TESTING_STARTING_ITERATIONS=60000
FINE_TESTING_STEP=5000
FINE_TESTING_MAX_ITERATIONS=60000
FINE_THRESHOLD=0.5
FINE_TIMESTAMP_X=_
FINE_TIMESTAMP_Y=_
FINE_TIMESTAMP_Z=_
COARSE2FINE_TESTING_MARGIN=$FINE_TRAINING_MARGIN
MAX_ROUNDS=10
# coarse2fine_testing.py : data_path, current_fold,
#     organ_number, low_range, high_range,
#     organ_ID, GPU_ID,
#     coarse_slice_thickness, coarse_learning_rate, coarse_gamma,
#     coarse_starting_iterations, coarse_step, coarse_max_iterations,
#     coarse_threshold,
#     coarse_timestamp_X (optional), coarse_timestamp_Y (optional), coarse_timestamp_Z (optional)
#     coarse_fusion_code, fine_slice_thickness, fine_training_margin,
#     fine_learning_rate, fine_gamma,
#     fine_starting_iterations, fine_step, fine_max_iterations,
#     fine_threshold,
#     fine_timestamp_X (optional), fine_timestamp_Y (optional), fine_timestamp_Z (optional)
#     coarse2fine_testing_margin, max_rounds
if [ "$ENABLE_COARSE2FINE_TESTING" = "1" ]
then
    python coarse2fine_testing.py \
        $DATA_PATH $CURRENT_FOLD $ORGAN_NUMBER $LOW_RANGE $HIGH_RANGE \
        $COARSE2FINE_TESTING_ORGAN_ID $COARSE2FINE_TESTING_GPU \
        $COARSE_SLICE_THICKNESS $COARSE_LEARNING_RATE $COARSE_GAMMA \
        $COARSE_TESTING_STARTING_ITERATIONS $COARSE_TESTING_STEP $COARSE_TESTING_MAX_ITERATIONS \
        $COARSE_THRESHOLD \
        $COARSE_TIMESTAMP_X $COARSE_TIMESTAMP_Y $COARSE_TIMESTAMP_Z \
        $COARSE_FUSION_CODE $FINE_SLICE_THICKNESS $FINE_TRAINING_MARGIN \
        $FINE_LEARNING_RATE $FINE_GAMMA \
        $FINE_TESTING_STARTING_ITERATIONS $FINE_TESTING_STEP $FINE_TESTING_MAX_ITERATIONS \
        $FINE_THRESHOLD \
        $FINE_TIMESTAMP_X $FINE_TIMESTAMP_Y $FINE_TIMESTAMP_Z \
        $COARSE2FINE_TESTING_MARGIN $MAX_ROUNDS
fi
