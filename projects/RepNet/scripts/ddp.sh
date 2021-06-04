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
from_path=$data_root/last5.pt
save_path=$data_root/last5.pt

# debug
sudo /bin/rm -rf /data/RepNet
cp /home/lidong/eta_codes /data/RepNet -aprf

docker exec -it raceai_base-test /data/RepNet/scripts/train.sh -n $rank/4 -b 12 -e 2000 -d $data_root -f $from_path -s $save_path
