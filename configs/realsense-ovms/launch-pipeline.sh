#!/bin/bash

# 1 - pipeline path
# 2 - camera serial number
# 3 - color_width
# 4 - color_height
# 5 - color_rate
# 6 - depth_width
# 7 - depth_height
# 8 - depth_rate
# 9 - frame_align ( 0-color_to_depth | 1-depth_to_color )
# 10 - enable_filters ( 0-none | 1-temporal | 2-hole_filling | 3-temporal and hole_filling )
# 11 - enable rendering 
# 12 - disable_rs_gpu_accel

echo "./$1 -- $2, $3, $4, $5, $6, $7, $8, $9, ${10}, ${11}, ${12}"
./$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}