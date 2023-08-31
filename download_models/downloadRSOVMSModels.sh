#!/bin/bash

# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#
# Todo: update document for how we get the bit model
# Here is detail about getting bit model: https://medium.com/openvino-toolkit/accelerate-big-transfer-bit-model-inference-with-intel-openvino-faaefaee8aec

REFRESH_MODE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --refresh)
            echo "running model downloader in refresh mode"
            REFRESH_MODE=1
            ;;
        *)
            echo "Invalid flag: $1" >&2
            exit 1
            ;;
    esac
    shift
done

MODEL_EXEC_PATH="$(dirname "$(readlink -f "$0")")"
modelDir="$MODEL_EXEC_PATH/../configs/realsense-ovms/models"
mkdir -p $modelDir
cd $modelDir || { echo "Failure to cd to $modelDir"; exit 1; }

if [ "$REFRESH_MODE" -eq 1 ]; then
    # cleaned up all downloaded files so it will re-download all files again
    rm yolov5s/1/*.xml || true; rm yolov5s/1/*.bin || true
    rm midasnet/1/*.xml || true; rm midasnet/1/*.bin || true
fi

# $1 model file name
# $2 download URL
# $3 model percision
# $4 local model folder name
getOVMSModelFiles() {
    # Make model directory
    mkdir -p $2
    mkdir -p $2/1
    
    # Get the models
    wget $1".bin" -P $2/1
    wget $1".xml" -P $2/1
}


if [ ! -f "yolov5s/1/yolov5s.xml" ]; then
    getOVMSModelFiles https://github.com/dlstreamer/pipeline-zoo-models/raw/main/storage/yolov5s-416_INT8/FP16-INT8/yolov5s yolov5s
fi

if [ ! -f "midasnet/1/midasnet.xml" ]; then
    omz_downloader --name midasnet -o midasnet/1/
    omz_converter --name midasnet --o midasnet/1/ -d midasnet/1/
    mv midasnet/1/public/midasnet/FP16/*.xml midasnet/1
    mv midasnet/1/public/midasnet/FP16/*.bin midasnet/1    
fi