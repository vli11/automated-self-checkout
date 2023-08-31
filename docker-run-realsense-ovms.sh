#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#


# Notes:
# Single JQ node update
#  jsonStr=`jq --arg name facedet --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
# All JQ nodes update
#  jsonStr=`jq --arg device GPU.0 '(.model_config_list[].config.target_device=$device)' <<< "$jsonStr"`

update_ovms_config_yolo() {

	ovms_jsonCfg=`cat configs/realsense-ovms/models/config_$PIPELINE_NAME.json`

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ] || [ "$HAS_FLEX_140" == "1" ]
		then
			ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`	
		else
			ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`	
		fi				
	fi
	echo $ovms_jsonCfg > configs/realsense-ovms/models/config_active.json
}


# clean up exited containers
docker rm $(docker ps -a -f name=realsense-ovms -f status=exited -q)

source benchmark-scripts/get-gpu-info.sh

if [ -z "$PLATFORM" ] || [ -z "$INPUTSRC" ] || [ -z "$PIPELINE_NAME" ]
then
	source get-options-realsense-demos.sh "$@"
fi

# This updates OVMS config files based on requested PLATFORM by user
# and then choosing the optimum device accerlation for the pipeline
if [ "$PIPELINE_NAME" == "object_size_estimation_yolov5" ]
then
	update_ovms_config_yolo
else
	echo "Unknown pipeline requested"
	exit 0
fi

# This sets the inference model caching dir for GPUs to load from (improves model loading time)
cl_cache_dir=`pwd`/.cl-cache
echo "CLCACHE: $cl_cache_dir"

TAG=openvino/model_server-capi-realsense:latest

if [ ! -z "$CONTAINER_IMAGE_OVERRIDE" ]
then
	echo "Using container image override $CONTAINER_IMAGE_OVERRIDE"
	TAG=$CONTAINER_IMAGE_OVERRIDE
fi

cids=$(docker ps  --filter="name=realsense-ovms" -q -a)
cid_count=`echo "$cids" | wc -w`
CONTAINER_NAME="realsense-ovms"$(($cid_count))
LOG_FILE_NAME="realsense-ovms"$(($cid_count))".log"

# Set RENDER_MODE=1 for demo purposes only
RUN_MODE="-itd"
if [ "$RENDER_MODE" == 1 ]
then
	RUN_MODE="-it"
	xhost +
fi

if [ -z "$DISABLE_RS_GPU_ACCEL" ]; then
	DISABLE_RS_GPU_ACCEL=0
fi

# cmdlineargs: 
# 1 - pipeline path
# 2 - camera serial number
# 3 - color_width
# 4 - color_height
# 5 - color_rate
# 6 - depth_width
# 7 - depth_height
# 8 - depth_rate
# 9 - frame_align ( 0-color_to_depth | 1-depth_to_color )
# 10 - enable_filters ( 00-none | 01-temporal | 10-hole_filling | 11-temporal and hole_filling )
# 11 - enable rendering 
# 12 - disable_rs_gpu_accel
bash_cmd="./launch-pipeline.sh $PIPELINE_EXEC_PATH $INPUTSRC $COLOR_WIDTH $COLOR_HEIGHT $COLOR_RATE $DEPTH_WIDTH $DEPTH_HEIGHT $DEPTH_RATE $FRAME_ALIGN $ENABLE_FITLERS $RENDER_MODE $DISABLE_RS_GPU_ACCEL"


# TODO Better to move this downloadModels to a container instead of clobbering the host
pip3 install openvino-dev onnx torchvision
# make sure models are downloaded or existing:
./download_models/getModels.sh --workload realsense-ovms

echo "BashCmd: $bash_cmd"
docker run --network host \
 $cameras $TARGET_USB_DEVICE $TARGET_GPU_DEVICE \
 --user root --ipc=host --name realsense-ovms$cid_count \
 $stream_density_mount \
 -e DISPLAY=$DISPLAY \
 -e cl_cache_dir=/home/intel/realsense-ovms/.cl-cache \
 -v $cl_cache_dir:/home/intel/realsense-ovms/.cl-cache \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v `pwd`/sample-media/:/home/intel/realsense-ovms/vids \
 -v `pwd`/configs/realsense-ovms/extensions:/home/intel/realsense-ovms/extensions \
 -v `pwd`/results:/tmp/results \
 -v `pwd`/configs/realsense-ovms/models:/home/intel/realsense-ovms/models \
 -w /home/intel/realsense-ovms \
 -e LOG_LEVEL=$LOG_LEVEL \
 -e cid_count=$cid_count \
  $RUN_MODE \
 $TAG "$bash_cmd"