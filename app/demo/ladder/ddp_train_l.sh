#!/bin/bash

pro=ladder
out=/raceai/data/tmp

master_addr=10.255.0.103
master_port=8555

# rm -rf /raceai/data/tmp/${pro}_l

python3 -m torch.distributed.launch \
	--nproc_per_node 1 --nnodes 3 --node_rank $DDP_RANK \
	--master_addr $master_addr --master_port $master_port \
	/raceai/codes/projects/yolov5/train.py \
	--img-size 640 --batch-size 60 --epochs 320  --device 0 \
	--workers 4 --project $out --name ${pro}_l --exist-ok \
	--data ./dataset.yaml \
	--hyp ./hyperparameters.yaml \
	--cfg ./yolov5l_small_achor.yaml \
	--weights /raceai/data/ckpts/yolov5/yolov5l.pt
#   --weights ""
#   --weights $out/${pro}_l/weights/last.pt
#   --weights /raceai/data/ckpts/yolov5/ladder/l.pt
