#!/bin/bash

project_root_dir=/project/train/src_repo
dataset_dir=/home/data
log_file=/project/train/log/log.txt

pip install -i https://mirrors.aliyun.com/pypi/simple -r /project/train/src_repo/requirements.txt \
&& echo "Preparing data..." \
&& python3 ${project_root_dir}/parse_data.py \
&& echo "Training..." \
&& python3 ${project_root_dir}/yolov5/train.py --batch 2 --epochs 30 --data ${project_root_dir}/config/wheat0.yaml --cfg ${project_root_dir}/yolov5/models/yolov5s.yaml --weights ${project_root_dir}/yolov5s.pt --name yolov5s_fold0 --img-size 512

#&& python3 ${project_root_dir}/yolov5/train.py --batch 2 --epochs 20 --data ${project_root_dir}/config/wheat0.yaml --cfg ${project_root_dir}/yolov5/models/yolov5m.yaml --#weights ${project_root_dir}/yolov5m.pt --name yolov5m_fold0 

