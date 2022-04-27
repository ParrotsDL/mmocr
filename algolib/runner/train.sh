#!/bin/bash
set -x

if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

if [ -x "$SMART_ROOT/submodules" ];then
    submodules_root=$SMART_ROOT
else    
    submodules_root=$PWD
fi

if [[ "$submodules_root" =~ "submodules/mmocr" ]]
then
    if [ ! -d "$submodules_root/../mmdet/mmdet" ]
    then
        cd ../..
        git submodule update --init submodules/mmdet
        cd -
    fi
else
    if [ ! -d "$submodules_root/submodules/mmdet/mmdet" ]
    then
        git submodule update --init submodules/mmdet
    fi
fi

# 0. placeholder
if [ -d "$submodules_root/submodules/mmocr/algolib/configs" ]
then
    rm -rf $submodules_root/submodules/mmocr/algolib/configs
    ln -s $submodules_root/submodules/mmocr/configs $submodules_root/submodules/mmocr/algolib/
else
    ln -s $submodules_root/submodules/mmocr/configs $submodules_root/submodules/mmocr/algolib/
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmocr/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules" ]]
then
    pyroot=$submodules_root/mmocr
else
    pyroot=$submodules_root/submodules/mmocr
fi
echo $pyroot
export PYTHONPATH=$pyroot:$PYTHONPATH
export FRAME_NAME=mmocr    #customize for each frame
export MODEL_NAME=$3

# mmdetpath
SHELL_PATH=$(dirname $0)
export PYTHONPATH=$SHELL_PATH/../../../mmdet:$PYTHONPATH

# init_path
export PYTHONPATH=$SMART_ROOT/common/sites/:$PYTHONPATH
 
# 4. build necessary parameter
partition=$1 
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}
 
# 5. model choice
export PARROTS_DEFAULT_LOGGER=FALSE


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
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

port=`expr $RANDOM % 10000 + 20000`

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
