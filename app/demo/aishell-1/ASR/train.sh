#!/bin/bash
#=================================================================
# date: 2021-03-29 16:05:20
# title: run
# author: QRS
#=================================================================

prog_name=$(basename $0)
master_addr=10.255.0.58
master_port=8555
nodes_num=1
procs_num=1
node_index=0
batch_size=3
ddp=False

__usage() {
    echo ""
    echo "$prog_name arguments:"
    echo "-m or --master master_addr:master_port"
    echo "-n or --nodes node/nnodes like 1/3"
    echo "-p or --procs num"
    echo "-h or --help"
    echo ""
}

ARGUMENT_LIST=(
    "master"
    "nodes"
    "procs"
)

opts=$(getopt \
    --options "dD$(printf "%.1s:" "${ARGUMENT_LIST[@]}")h::" \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")help::" \
    --name "$prog_name" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|-D)
            ddp=True
            shift 1
            ;;

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

        -h|--help)
            __usage && exit 1
            ;;

        *)
            break
            ;;
    esac
done

batch_size=`expr $batch_size \* $nodes_num`
output_dir=/data/tmp/sb/aishell-1
data_root=/data/datasets/asr

commargs="--batch_size $batch_size \
    --output_folder $output_dir \
    --data_folder ${data_root}/AISHELL-1/ \
    --data_folder_rirs ${data_root}/noises/ \
    --tokenizer_file /data/pretrained/asr/aishell-1/tokenizer/5000_unigram.model"

pid=`ps -eo pid,args | grep "train.py" | grep -v "grep" | cut -c 1-6`
if [[ x$pid != x ]]
then
    echo "kill $pid"
    kill -9 $pid
fi

# rm -rf $output_dir/save

if [[ x$ddp == xTrue ]]
then
    echo "DDP run $batch_size"
    python3 -m torch.distributed.launch \
        --nproc_per_node=$procs_num --nnodes=$nodes_num --node=$node_index \
        --master_addr $master_addr --master_port $master_port \
        train.py hparams/train.yaml \
        --distributed_launch=True --distributed_backend='nccl' --auto_mix_prec=True \
        $commargs
else
    python3 train.py hparams/train.yaml $commargs
fi
