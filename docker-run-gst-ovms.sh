#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#



# Below variable is used for choosing which GPU to use for media acclerated decode/encode.
# Examples: 
#  Serpent Canyon reference:  D128 is the SOC iGPU vs. D129 is dGPU
#  For Xeon Servers (Gold/Silver/Platinum) D128 is first dGPU, D129 is the second, etc
export GST_VAAPI_DRM_DEVICE=/dev/dri/renderD128

update_media_device_engine() {
	# Use discrete GPU if it exists, otherwise use iGPU or CPU
	if [ "$PLATFORM" != "cpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_140" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			export GST_VAAPI_DRM_DEVICE=/dev/dri/renderD129
		fi
	fi
}

# Notes:
# Single JQ node update
#  jsonStr=`jq --arg name facedet --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
# All JQ nodes update
#  jsonStr=`jq --arg device GPU.0 '(.model_config_list[].config.target_device=$device)' <<< "$jsonStr"`

update_ovms_config_face_detection() {

	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		TARGET_GPU_DEVICE="--privileged"

		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			TARGET_GPU="GPU.1"
			ovms_jsonCfg=`jq --arg name face_detection --arg device GPU.1 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$ovms_jsonCfg"`
			ovms_jsonCfg=`jq --arg name face_landmarks --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$ovms_jsonCfg"`
		elif [ "$HAS_FLEX_140" == "1" ] 
		then
			TARGET_GPU="GPU.0"
			ovms_jsonCfg=`jq --arg name face_detection --arg device GPU.0 '(.model_config_list |= map(select(.config.name == "face_detection").config.target_device=$device))' <<< "$ovms_jsonCfg"`
			ovms_jsonCfg=`jq --arg name face_landmarks --arg device CPU '( .model_config_list |= map(select(.config.name == "face_landmarks").config.target_device=$device))' <<< "$ovms_jsonCfg"`
		else
			TARGET_GPU="GPU.0"
			ovms_jsonCfg=`jq --arg name face_detection --arg device GPU.0 '(.model_config_list |= map(select(.config.name == "face_detection").config.target_device=$device))' <<< "$ovms_jsonCfg"`
			ovms_jsonCfg=`jq --arg name face_landmarks --arg device CPU '( .model_config_list |= map(select(.config.name == "face_landmarks").config.target_device=$device))' <<< "$ovms_jsonCfg"`
		fi				
	fi
	echo $ovms_jsonCfg
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}

update_ovms_config_object_and_text_detection() {

	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`		

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			ovms_jsonCfg=`jq --arg name yolov5s --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name text-detect-0002 --arg device GPU.1 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		elif [ "$HAS_FLEX_140" == "1" ] 
		then
			ovms_jsonCfg=`jq --arg name yolov5s --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name text-detect-0002 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		else
			ovms_jsonCfg=`jq --arg name yolov5s --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name text-detect-0002 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		fi				
	fi
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}

update_ovms_config_people_detection() {
    
	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`		

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		# Keeping seperate from GPU so that if  > 1 models such as people reid / attribute reid
		# models are used we can XPU load them
		# Since only one model is used for people detection today behavior is the same as GPU
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	fi
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}

update_ovms_config_geti_yolox() {

	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.1 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		elif [ "$HAS_FLEX_140" == "1" ] 
		then
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		else
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		fi				
	fi
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}

update_ovms_config_geti_yolox_ensemble() {

	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.1 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		elif [ "$HAS_FLEX_140" == "1" ] 
		then
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		else
			ovms_jsonCfg=`jq --arg name geti_yolox --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		fi				
	fi
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}

update_ovms_config_yolov5_ensemble() {

	ovms_jsonCfg=`cat configs/gst-ovms/models/config_$PIPELINE_NAME.json`

	if [ "$PLATFORM" == "gpu" ]
	then
		ovms_jsonCfg=`jq --arg device $TARGET_GPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`

	elif [ "$PLATFORM" == "cpu" ]
	then
		ovms_jsonCfg=`jq --arg device CPU '(.model_config_list[].config.target_device=$device)' <<< "$ovms_jsonCfg"`
	elif [ "$PLATFORM" == "xpu" ]
	then
		if [ "$HAS_ARC" == "1" ] || [ "$HAS_FLEX_170" == "1" ]
		then
			ovms_jsonCfg=`jq --arg name yolov5 --arg device GPU.1 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		elif [ "$HAS_FLEX_140" == "1" ] 
		then
			ovms_jsonCfg=`jq --arg name yolov5 --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		else
			ovms_jsonCfg=`jq --arg name yolov5 --arg device GPU.0 '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
			ovms_jsonCfg=`jq --arg name efficientnetb0 --arg device CPU '( .model_config_list[] | select(.config.name == $name) ).device |= $device' <<< "$jsonStr"`
		fi				
	fi
	echo $ovms_jsonCfg > configs/gst-ovms/models/config_active.json
}


# clean up exited containers
docker rm $(docker ps -a -f name=gst-ovms -f status=exited -q)

#export GST_DEBUG=0
if [ -z "$USE_ONEVPL" ]; then
	USE_ONEVPL=0
fi

if [ -z "$RENDER_PORTRAIT_MODE" ]; then
	RENDER_PORTRAIT_MODE=0
fi


source benchmark-scripts/get-gpu-info.sh

if [ -z "$PLATFORM" ] || [ -z "$INPUTSRC" ] || [ -z "$PIPELINE_NAME" ]
then
	source get-options-gst-demos.sh "$@"
fi

# This updates OVMS config files based on requested PLATFORM by user
# and then choosing the optimum device accerlation for the pipeline
if [ "$PIPELINE_NAME" == "face_detection" ]
then
	update_ovms_config_face_detection

elif [ "$PIPELINE_NAME" == "object_and_text_detection" ]
then
	update_ovms_config_object_and_text_detection

elif [ "$PIPELINE_NAME" == "people_detection" ]
then
	update_ovms_config_people_detection

elif [ "$PIPELINE_NAME" == "geti_yolox" ]
then

	update_ovms_config_geti_yolox

elif [ "$PIPELINE_NAME" == "geti_yolox_ensemble" ]
then

	update_ovms_config_geti_yolox_ensemble

elif [ "$PIPELINE_NAME" == "yolov5_ensemble" ]
then

	update_ovms_config_yolov5_ensemble

else
	echo "Unknown pipeline requested"
	exit 0
fi

# This updates the media GPU engine utilized based on the request PLATFOR by user
# The default state of all libva (*NIX) media decode/encode/etc is GPU.0 instance
update_media_device_engine

# This sets the inference model caching dir for GPUs to load from (improves model loading time)
cl_cache_dir=`pwd`/.cl-cache
echo "CLCACHE: $cl_cache_dir"

TAG=openvino/model_server-capi-gst:latest

if [ ! -z "$CONTAINER_IMAGE_OVERRIDE" ]
then
	echo "Using container image override $CONTAINER_IMAGE_OVERRIDE"
	TAG=$CONTAINER_IMAGE_OVERRIDE
fi

cids=$(docker ps  --filter="name=gst-ovms" -q -a)
cid_count=`echo "$cids" | wc -w`
CONTAINER_NAME="gst-ovms"$(($cid_count))
LOG_FILE_NAME="gst-ovms"$(($cid_count))".log"

# Set RENDER_MODE=1 for demo purposes only
RUN_MODE="-itd"
if [ "$RENDER_MODE" == 1 ]
then
	RUN_MODE="-it"
	xhost +
fi

if grep -q "file" <<< "$INPUTSRC"; then
	# filesrc	
	arrfilesrc=(${INPUTSRC//:/ })
	# use vids since container maps a volume to this location based on sample-media folder
	INPUTSRC="./vids/"${arrfilesrc[1]}
fi

# cmdlineargs: inputsrc 0-libva|1-onevpl 0-norender|1-render
bash_cmd="./launch-pipeline.sh $PIPELINE_EXEC_PATH $INPUTSRC $USE_ONEVPL $RENDER_MODE $RENDER_PORTRAIT_MODE"

if [ "$STREAM_DENSITY_MODE" == 1 ]; then
	echo "Starting Stream Density"
	bash_cmd="./stream_density_framework-pipelines.sh framework-pipelines/$PLATFORM/$pipeline"
	stream_density_mount="-v `pwd`/configs/dlstreamer/framework-pipelines/stream_density.sh:/home/pipeline-server/stream_density_framework-pipelines.sh"
	stream_density_params="-e STREAM_DENSITY_FPS=$STREAM_DENSITY_FPS -e STREAM_DENSITY_INCREMENTS=$STREAM_DENSITY_INCREMENTS -e COMPLETE_INIT_DURATION=$COMPLETE_INIT_DURATION"
	echo "DEBUG: $stream_density_params"
fi

#echo "DEBUG: $TARGET_GPU_DEVICE $PLATFORM $HAS_FLEX_140"
if [ "$TARGET_GPU_DEVICE" == "--privileged" ] && [ "$PLATFORM" == "dgpu" ] && [ $HAS_FLEX_140 == 1 ]
then
	if [ "$STREAM_DENSITY_MODE" == 1 ]; then
		# override logic in workload script so stream density can manage it
		AUTO_SCALE_FLEX_140=2
	else
		# allow workload to manage autoscaling
		AUTO_SCALE_FLEX_140=1
	fi
fi

# make sure models are downloaded or existing:
./download_models/getModels.sh --workload gst-ovms

# -v `pwd`/configs/gst-ovms/pipelines:/home/intel/gst-ovms/pipelines \
echo "BashCmd: $bash_cmd with media on $GST_VAAPI_DRM_DEVICE with USE_ONEVPL=$USE_ONEVPL"
docker run --network host \
 $cameras $TARGET_USB_DEVICE $TARGET_GPU_DEVICE \
 --user root --ipc=host --name gst-ovms$cid_count \
 $stream_density_mount \
 -e DISPLAY=$DISPLAY \
 -e GST_VAAPI_DRM_DEVICE=$GST_VAAPI_DRM_DEVICE \
 -e cl_cache_dir=/home/intel/gst-ovms/.cl-cache \
 -v $cl_cache_dir:/home/intel/gst-ovms/.cl-cache \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v `pwd`/sample-media/:/home/intel/gst-ovms/vids \
 -v `pwd`/configs/gst-ovms/extensions:/home/intel/gst-ovms/extensions \
 -v `pwd`/results:/tmp/results \
 -v `pwd`/configs/gst-ovms/models:/home/intel/gst-ovms/models \
 -w /home/intel/gst-ovms \
 -e LOG_LEVEL=$LOG_LEVEL \
 -e GST_DEBUG=$GST_DEBUG \
 -e RENDER_MODE=$RENDER_MODE \
 -e INPUTSRC_TYPE=$INPUTSRC_TYPE \
 -e inputsrc="$inputsrc" \
 -e decode_type="$decode_type" \
 -e USE_ONEVPL="$USE_ONEVPL" \
 -e cid_count=$cid_count \
 -e RENDER_PORTRAIT_MODE=$RENDER_PORTRAIT_MODE \
 -e AUTO_SCALE_FLEX_140="$AUTO_SCALE_FLEX_140" \
 $RUN_MODE $stream_density_params \
 $TAG "$bash_cmd"