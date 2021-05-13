set -x

# master_addr=10.255.0.58
master_addr=10.255.0.104
master_port=8555
nnodes=4

pids=`ps -eo pid,args | grep "train.py" | grep -v grep | cut -c1-5`

echo $pids

export PYTHONPATH=/data/RepNet:/hzcsk12/hzcsnote

cur_fil=${BASH_SOURCE[0]}
cur_dir=`dirname $cur_fil`
top_dir=`dirname $cur_dir`

if [[ x$pids != x ]]
then
    kill -9 $pids
fi

num_epochs=200
batch_size=8

if [[ x$1 != x ]]
then
	rank=$1
	if [[ x$2 != x ]]
	then
		num_epochs=$2
		shift
	fi
	if [[ x$2 != x ]]
	then
		batch_size=$2
	fi
	python3 -m torch.distributed.launch \
		--nproc_per_node=1 --nnodes=$nnodes --node_rank=$rank \
		--master_addr $master_addr --master_port $master_port \
		$top_dir/repnet/train.py \
		--num_epochs $num_epochs \
		--batch_size $batch_size
else
	python3 $top_dir/repnet/train.py
fi
