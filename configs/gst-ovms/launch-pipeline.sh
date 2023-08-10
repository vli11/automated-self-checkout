#!/bin/bash

# 1 - pipeline path
# 2 - inputsrc
# 3 - use_onevpl
# 4 - enable rendering 
# 5 - codec_type (avc or hevc)

source ./get-media-codec.sh $2
echo "./$1 $2 $3 $4 $is_avc"
./$1 $2 $3 $4 $is_avc