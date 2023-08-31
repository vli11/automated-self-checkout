#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

REALSENSE_ENABLED=0
PIPELINE_NAME=
PIPELINE_EXEC_PATH=
PLATFORM=

error() {
    printf '%s\n' "$1" >&2
    exit 1
}

show_help() {
	echo " Under development (benchmarking is not validated and likely many other features)
         usage: ./docker-run.sh --platform gpu|cpu|xpu --inputsrc CAMERA_RTSP_URL|file:video.mp4|/dev/video0 [--realsense_enabled] --pipeline face_detection|people_detection_retail|people_detection_yolov5s|object_and_text_detection|geti_yolox

         Note: 
	  1. gpu.x should be replaced with targeted GPUs such as gpu (for all GPUs), gpu.0, gpu.1, etc
	  2. filesrc will utilize videos stored in the sample-media folder
	  3. Set environment variable STREAM_DENSITY_MODE=1 for starting single container stream density testing
	  4. Set environment variable RENDER_MODE=1 for displaying pipeline and overlay CV metadata
	  5. Set environment variable USE_ONEVPL=1 for oneVPL
	  6. Set environment variable RENDER_PORTRAIT_MODE=1 for rendering with portrait mode
        "
}

while :; do
    case $1 in
    -h | -\? | --help)
        show_help
        exit
        ;;
    --platform)
        if [ "$2" ]; then
            if [ $2 == "cpu" ]; then
                PLATFORM=$2
                shift
            elif [ $2 == "xpu" ]; then
                PLATFORM=$2
                shift
            elif grep -q "gpu" <<< "$2"; then
                PLATFORM="gpu"
                arrgpu=(${2//./ })
                TARGET_GPU_NUMBER=${arrgpu[1]}
                if [ -z "$TARGET_GPU_NUMBER" ]; then
                    TARGET_GPU="GPU.0"
                    TARGET_GPU_DEVICE="--privileged"
                else
                    TARGET_GPU_ID=$((128+$TARGET_GPU_NUMBER))
                    TARGET_GPU="GPU."$TARGET_GPU_NUMBER
                    #TARGET_GPU_DEVICE="--device=/dev/dri/renderD"$TARGET_GPU_ID
                    #TARGET_GPU_DEVICE_NAME="/dev/dri/renderD"$TARGET_GPU_ID
                    TARGET_GPU_DEVICE="--privileged"
                fi
                shift
            else
                error 'ERROR: "--platform" requires an argument gpu|cpu|xpu.'
            fi
        else
                error 'ERROR: "--platform" requires an argument gpu|cpu|xpu.'
        fi	    
        ;;
    --inputsrc)
        if [ "$2" ]; then
            INPUTSRC=$2
            shift
        else
            error 'ERROR: "--inputsrc" requires an argument RS_SERIAL_NUMBER|CAMERA_RTSP_URL|file:video.mp4|/dev/video0.'
        fi
        ;;
    --realsense_enabled)
        REALSENSE_ENABLED=1
        ;;
    --pipeline)
	    if [ "$2" ]; then
	        PIPELINE_NAME="${2}"
            PIPELINE_EXEC_PATH="pipelines/${2}/${2}"
		    shift
	    else
            error 'ERROR: "--pipeline" requires an argument face_detection|people_detection_retail|people_detection_yolov5s|object_and_text_detection|geti_yolox'
	    fi
	    ;;
    -?*)
        error "ERROR: Unknown option $1"
        ;;
    ?*)
        error "ERROR: Unknown option $1"
        ;;
    *)
        break
        ;;
    esac

    shift

done

if [ -z $PLATFORM ] || [ -z $INPUTSRC ] || [ -z $PIPELINE_NAME ]
then
	#echo "Blanks: $1 $PLATFORM $INPUTSRC"
   	show_help
	exit 0
fi