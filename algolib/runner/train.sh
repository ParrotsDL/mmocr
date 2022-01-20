#!/bin/bash
set -x

workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmocr" ]]
then
    if [ ! -d "$workdir/../mmdet/mmdet" ]
    then
        cd ../..
        git submodule update --init submodules/mmdet
        cd -
    fi
else
    if [ ! -d "$workdir/submodules/mmdet/mmdet" ]
    then
        git submodule update --init submodules/mmdet
    fi
fi

# 0. placeholder
workdir=$(cd $(dirname $1); pwd)
if [[ "$workdir" =~ "submodules/mmocr" ]]
then
    if [ -d "$workdir/algolib/configs" ]
    then
        rm -rf $workdir/algolib/configs
        ln -s $workdir/configs $workdir/algolib/
    else
        ln -s $workdir/configs $workdir/algolib/
    fi
else
    if [ -d "$workdir/submodules/mmocr/algolib/configs" ]
    then
        rm -rf $workdir/submodules/mmocr/algolib/configs
        ln -s $workdir/submodules/mmocr/configs $workdir/submodules/mmocr/algolib/
    else
        ln -s $workdir/submodules/mmocr/configs $workdir/submodules/mmocr/algolib/
    fi
fi
 
# 1. build file folder for save log,format: algolib_gen/frame
mkdir -p algolib_gen/mmocr/$3
export PYTORCH_VERSION=1.4
 
# 2. set time
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env
path=$PWD
if [[ "$path" =~ "submodules/mmocr" ]]
then
    pyroot=$path
    comroot=$path/../..
    init_path=$path/..
else
    pyroot=$path/submodules/mmocr
    comroot=$path
    init_path=$path/submodules
fi
echo $pyroot
export PYTHONPATH=$comroot:$pyroot:$PYTHONPATH
export FRAME_NAME=mmocr    #customize for each frame
export MODEL_NAME=$3
 
# mmcv path
version_p=$(python -c 'import sys; print(sys.version_info[:])')
CONDA_ROOT=/mnt/cache/share/platform/env/miniconda3.${version_p:4:1}
MMCV_PATH=${CONDA_ROOT}/envs/${CONDA_DEFAULT_ENV}/mmcvs
mmcv_version=1.3.15
export PYTHONPATH=${MMCV_PATH}/${mmcv_version}:$PYTHONPATH
export MMCV_HOME=/mnt/lustre/share_data/parrots_algolib/datasets/pretrain/mmcv

# mmdetpath
SHELL_PATH=$(dirname $0)
export PYTHONPATH=$PYTHONPATH:$SHELL_PATH/../../../mmdet

# init_path
export PYTHONPATH=$init_path/common/sites/:$PYTHONPATH # necessary for init
 
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

set -x

file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} --options dist_params.port=$port $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
