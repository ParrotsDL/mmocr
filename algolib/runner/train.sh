#!/bin/bash
set -x
set -o pipefail

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmocr" ]]
then 
    pyroot=$PWD
    if [ ! -d "$PWD/../mmdet/mmdet" ]
    then
        cd ..
        git submodule update --init mmdet
        cd -
    fi
else
    pyroot=$PWD/mmocr
    if [ ! -d "$PWD/mmdet/mmdet" ]
    then
        git submodule update --init mmdet
    fi
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi
 
# 2. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmocr/$3
now=$(date +"%Y%m%d_%H%M%S")

# 3. set env variables
export PYTORCH_VERSION=1.4
export PYTHONPATH=$pyroot:$PYTHONPATH
export FRAME_NAME=mmocr    #customize for each frame
export MODEL_NAME=$3
export PARROTS_DEFAULT_LOGGER=FALSE

# 4. mmdetpath and init_path
export PYTHONPATH=$pyroot/../mmdet:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH
 
# 5. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 6. model list
case $MODEL_NAME in
    "dbnet_r50dcnv2_fpnc_1200e_icdar2015")
        FULL_MODEL="textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015"
        ;;
    "panet_r18_fpem_ffm_600e_icdar2015")
        FULL_MODEL="textdet/panet/panet_r18_fpem_ffm_600e_icdar2015"
        ;;
    "crnn_academic_dataset")
        FULL_MODEL="textrecog/crnn/crnn_academic_dataset"
        ;;
    "satrn_academic")
        FULL_MODEL="textrecog/satrn/satrn_academic"
        ;;
    "sar_r31_parallel_decoder_academic")
        FULL_MODEL="textrecog/sar/sar_r31_parallel_decoder_academic"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
