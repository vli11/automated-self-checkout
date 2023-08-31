#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

REFRESH_INPUT=
OPEN_OVMS=0
GST_OVMS=0
RS_OVMS=0

show_help() {
    echo "
        usage: $0
            --workload dlstreamer|opencv-ovms|gst-ovms
            --refresh 
        Note:
            1. --refresh is optional, it removes previously downloaded model files
            2. --workload is default to dlstreamer if not provided 
    "
}

get_options() {
    while :; do
        case $1 in
            -h | -\? | --help)
                show_help
                exit 0
            ;;
            --refresh)
                echo "running model downloader in refresh mode"
                REFRESH_INPUT="--refresh"
                ;;
            --workload)
                echo "workload: ${2}"
                if [ "$2" == "gst-ovms" ]; then
                    GST_OVMS=1
                elif [ "$2" == "realsense-ovms" ]; then
                    RS_OVMS=1
                elif [ "$2" == "opencv-ovms" ]; then
                    OPEN_OVMS=1
                else 
                    if [ "$2" != "dlstreamer" ]; then
                        echo 'ERROR: "--workload" requires an argument dlstreamer|opencv-ovms|gst-ovms'
                        exit 1
                    fi
                fi
                shift
                ;;
            *)
                break
                ;;
            esac
            shift
        done

}

if [ -z $1 ]
then
    show_help
fi

get_options "$@"

MODEL_EXEC_PATH="$(dirname "$(readlink -f "$0")")"
echo $MODEL_EXEC_PATH
if [ "$OPEN_OVMS" -eq 1 ]; then
    echo "Starting open-ovms model download..."
    $MODEL_EXEC_PATH/downloadOVMSModels.sh $REFRESH_INPUT
elif [ "$RS_OVMS" -eq 1 ]; then
    echo "Starting realsense-ovms model download..."
    $MODEL_EXEC_PATH/downloadRSOVMSModels.sh $REFRESH_INPUT
elif [ "$GST_OVMS" -eq 1 ]; then
    echo "Starting gst-ovms model download..."
    $MODEL_EXEC_PATH/downloadGSTOVMSModels.sh $REFRESH_INPUT
else
    echo "Starting dlstreamer model download..."
    $MODEL_EXEC_PATH/modelDownload.sh $REFRESH_INPUT
fi
