#!/bin/bash

# 检查传入的参数数量
if [ $# -ne 4 ]; then
    echo "Usage: $0 <00> <00>  <test_file_path> <param_set>"
    exit 1
fi

# 传入的参数
param="$1"
out_name="$2"
test_pose_path="$3"
param2="$4"
# 选择相应的文件地址 
input_file="/home/orangepi/Downloads/posetrue_kitti_to_tum/kitti_${param}_tum.txt"
# 执行命令
evo_ape tum "$input_file" "$test_pose_path" "$param2" -r full --plot --plot_mode xz --save_results "result/kitti${param}_${out_name}.zip" --save_plot "result/kitti${param}_${out_name}.eps"
