#!/bin/bash

# set -x

export PYTHONPATH=/data/RepNet:/hzcsk12/hzcsnote

prog_name=$(basename $0)
cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`
top_dir=`dirname $cur_dir`

# master_addr=10.255.0.58
master_addr=10.255.0.101
master_port=8555
nodes_num=1
procs_num=1
node_index=0
batch_size=12
num_epochs=2000
ddp=False
from_path=/tmp/last.pt
save_path=/tmp/last.pt
data_root=/data/datasets/cv/countix

__usage() {
    echo ""
    echo "${prog_name} arguments:"
    echo "-m or --master master_addr:master_port"
    echo "-n or --nodes node/nnodes like 1/3"
    echo "-p or --procs num"
    echo "-b or --bs batchsize"
    echo "-e or --epochs num epochs"
    echo "-d or --data_root  dataset root path"
    echo "-f or --fromtag ckpt from"
    echo "-s or --savetag ckpt save"
    echo "-h or --help"
    echo ""
}

ARGUMENT_LIST=(
    "master"
    "nodes"
    "procs"
    "epochs"
    "data_root"
    "from_path"
    "save_path"
    "bs"
)

opts=$(getopt \
    --options "$(printf "%.1s:" "${ARGUMENT_LIST[@]}")h::" \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")help::" \
    --name "$prog_name" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--master)
            ddp=True
            master_addr=`echo $2 | cut -d: -f1`
            master_port=`echo $2 | cut -d: -f2`
            shift 2
            ;;

        -n|--nodes)
            ddp=True
            nodes_num=`echo $2 | cut -d/ -f2`
            node=`echo $2 | cut -d/ -f1`
            node_index=`expr $node - 1`
            shift 2
            ;;

        -p|--procs)
            ddp=True
            procs_num=$2
            shift 2
            ;;

        -b|--bs)
            batch_size=$2
            shift 2
            ;;

        -e|--epochs)
            num_epochs=$2
            shift 2
            ;;

        -d|--data_root)
            data_root=$2
            shift 2
            ;;

        -f|--from_path)
            from_path=$2
            shift 2
            ;;

        -s|--save_path)
            save_path=$2
            shift 2
            ;;

        -h|--help)
            __usage && exit 1
            ;;

        *)
            break
            ;;
    esac
done

# batch_size=`expr $batch_size \* $nodes_num`

__kill_resource() {
    pid=`ps -eo pid,args | grep "train.py" | grep -v "grep" | cut -c 1-6`
    if [[ x$pid != x ]]
    then
        echo "kill $pid"
        kill -9 $pid
    fi
}

__kill_resource

if [[ x$ddp == xTrue ]]
then
	cmd="python3 -m torch.distributed.launch \
        --nproc_per_node=$procs_num --nnodes=$nodes_num --node=$node_index \
		--master_addr $master_addr --master_port $master_port \
        $top_dir/repnet/train.py \
        --num_epochs $num_epochs \
        --batch_size $batch_size \
        --data_root $data_root \
        --ckpt_from_path $from_path \
        --ckpt_save_path $save_path"
    echo $cmd
    $cmd
else
	python3 $top_dir/repnet/train.py
fi
