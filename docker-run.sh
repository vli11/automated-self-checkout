#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

source benchmark-scripts/get-gpu-info.sh
# WORKLOAD_SCRIPT is env varilable will be overwritten by --workload input option


WORKLOAD_SCRIPT=$WORKLOAD_SCRIPT
if [ -z $WORKLOAD_SCRIPT ]; then
    WORKLOAD_SCRIPT="docker-run-dlstreamer.sh"
fi

echo "running $WORKLOAD_SCRIPT"
./$WORKLOAD_SCRIPT "$@"
