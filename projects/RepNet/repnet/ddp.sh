#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

hostname=$(hostname)

rank=${hostname: -1}

from_path=1
save_path=1

if [[ x$1 != x ]]
then
    from_path=$1
    save_path=$2
fi

data_root=/data/datasets/cv/countix
from_path=$data_root/last.pt
save_path=$data_root/last.pt

# debug
cp /home/lidong/eta_data/RepNet/repnet/train.py $CUR_DIR/../repnet/train.py
cp /home/lidong/eta_data/RepNet/repnet/models/repnet.py $CUR_DIR/../repnet/models/repnet.py

docker exec -it raceai_base-test $CUR_DIR/train.sh -n $rank/4 -b 12 -e 2000 -d $data_root -f $from_path -s $save_path
