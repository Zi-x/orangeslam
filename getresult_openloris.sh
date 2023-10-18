#!/bin/bash

# 检查传入的参数数量
if [ $# -ne 4 ]; then
    echo "Usage: $0 <datasetname> <outname>  <test_file_path> <evoparam>"
    exit 1
fi

# 传入的参数
param="$1"
out_name="$2"
test_pose_path="$3"
param2="$4"
# 选择相应的文件地址 
input_file="/home/orangepi/Downloads/Dataset/openloris/${param}/groundtruth.txt"
# 执行命令
evo_ape tum "$input_file" "$test_pose_path" "$param2" -r full --plot --plot_mode xy --save_results "result/openloris${param}_${out_name}.zip" --save_plot "result/openloris${param}_${out_name}.eps"
