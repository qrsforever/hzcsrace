#!/bin/bash
#=================================================================
# date: 2021-03-29 16:05:20
# title: run
# author: QRS
#=================================================================

prog_name=$(basename $0)
master_addr=10.255.0.58
master_port=8555
nodes_num=2
procs_num=1
node_index=0
ddp=False
commargs="--output_folder /data/tmp/sb/aishell-1 \
    --data_folder /data/datasets/asr/AISHELL-1/ \
    --data_folder_rirs /data/datasets/asr/noises/ \
    --tokenizer_file /data/pretrained/asr/aishell-1/tokenizer/5000_unigram.model"

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
    "help"
)

opts=$(getopt \
    --options "dD$(printf "%.1s:" "${ARGUMENT_LIST[@]}")" \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
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
            master_addr=`echo $2 | cut -d: -f1`
            master_port=`echo $2 | cut -d: -f2`
            ddp=True
            shift 2
            ;;

        -n|--nodes)
            nodes_num=`echo $2 | cut -d/ -f2`
            node=`echo $2 | cut -d/ -f1`
            node_index=`expr $node - 1`
            shift 2
            ;;

        -p|--procs)
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

if [[ x$ddp == xTrue ]]
then
    python3 -m torch.distributed.launch \
        --nproc_per_node=$procs_num --nnodes=$nodes_num --node=$node_index \
        --master_addr $master_addr --master_port $master_port \
        train.py hparams/train.yaml \
        --distributed_launch=True --distributed_backend='nccl' \
        $commargs
else
    python3 train.py hparams/train.yaml $commargs
fi
