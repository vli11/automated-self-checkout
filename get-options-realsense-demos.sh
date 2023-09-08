#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#


PIPELINE_NAME=
PIPELINE_EXEC_PATH=
PLATFORM=
COLOR_WIDTH=0
COLOR_HEIGHT=0
COLOR_RATE=0
DEPTH_WIDTH=0
DEPTH_HEIGHT=0
DEPTH_RATE=0
FRAME_ALIGN=0
ENABLE_FITLERS=0

error() {
    printf '%s\n' "$1" >&2
    exit 1
}

show_help() {
	echo " 
         usage: ./docker-run.sh --platform gpu.x|cpu|xpu --inputsrc CAMERA_SERIAL_NUMBER or AUTO --pipeline object_size_estimation_yolov5 [ --color_width 1280 --color_height 720 --color_rate 15 --depth_width 1280 --depth_height 720 --depth_rate 30 --frame_alignment 0|1 --enable_filters 0|1|2|3 ]

         Note: 
	  1. gpu.x should be replaced with targeted GPUs such as gpu (for all GPUs), gpu.0, gpu.1, etc
	  2. frame_alignment should be set to 0  none, 1 for depth_to_color, 2 color_to_depth
	  3. enable_filters should be set to 0 for none, 1 for temporal, 2 for hole_filling, 3 for temporal and hole_filling
	  4. Set environment variable DISABLE_RS_GPU_ACCEL=1 for using CPU only for realsense operations (frame alignment/etc)
	  5. Set environment variable RENDER_MODE=1 for displaying pipeline and overlay CV metadata
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
            error 'ERROR: "--inputsrc" requires an argument CAMERA_SERIAL_NUMBER or AUTO'
        fi
        ;;
    --pipeline)
	    if [ "$2" ]; then
	        PIPELINE_NAME="${2}"
            PIPELINE_EXEC_PATH="pipelines/${2}/${2}"
		    shift
	    else
            error 'ERROR: "--pipeline" requires an argument object_size_estimation_yolov5'
	    fi
	    ;;
    --color_width)
        if [ "$2" ]; then
	        COLOR_WIDTH="${2}"
		    shift
	    #else
        #    error 'ERROR: "--color_width" requires an argument'
	    fi
        ;;
    --color_height)
        if [ "$2" ]; then
	        COLOR_HEIGHT="${2}"
		    shift
	    #else
        #    error 'ERROR: "--color_height" requires an argument'
	    fi
        ;;
    --color_rate)
        if [ "$2" ]; then
	        COLOR_RATE="${2}"
		    shift
	    #else
        #    error 'ERROR: "--color_rate" requires an argument'
	    fi
        ;;
    --depth_width)
        if [ "$2" ]; then
	        DEPTH_WIDTH="${2}"
		    shift
	    #else
        #    error 'ERROR: "--depth_width" requires an argument'
	    fi
        ;;
    --depth_height)
        if [ "$2" ]; then
	        DEPTH_HEIGHT="${2}"
		    shift
	    #else
        #    error 'ERROR: "--depth_height" requires an argument'
	    fi
        ;;
    --depth_rate)
        if [ "$2" ]; then
	        DEPTH_RATE="${2}"
		    shift
	    #else
        #    error 'ERROR: "--depth_rate" requires an argument'
	    fi
        ;;
    --frame_alignment)
        # 0|1|2
        if [ "$2" ]; then
	        FRAME_ALIGN="${2}"
		    shift
	    #else
        #    error 'ERROR: "--frame_alignment" requires an argument'
	    fi
        ;;
    --enable_filters)
        # 00|01|10|11
        if [ "$2" ]; then
	        ENABLE_FITLERS="${2}"
		    shift
	    #else
        #    error 'ERROR: "--enable_filters" requires an argument'
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
   	show_help
	exit 0
fi