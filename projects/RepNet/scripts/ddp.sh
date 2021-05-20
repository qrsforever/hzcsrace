#!/bin/bash

CUR_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

hostname=$(hostname)

rank=${hostname: -1}

from_tag=1
save_tag=1

if [[ x$1 != x ]]
then
    from_tag=$1
    save_tag=$2
fi

docker exec -it raceai_base-test $CUR_DIR/train.sh -n $rank/4 -b 12 -e 2000 -f $from_tag -s $save_tag
