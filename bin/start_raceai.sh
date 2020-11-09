#!/bin/bash
#=================================================================
# date: 2020-11-09 13:11:39
# title: start_raceai
# author: QRS
#=================================================================

cur_fil=${BASH_SOURCE[0]}
top_dir=`cd $(dirname $cur_fil)/..; pwd`

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse HEAD | cut -c 1-12)
NUMBER=$(git rev-list HEAD | wc -l | awk '{print $1}')

ifnames=("eth0" "ens3")
__find_lanip() {
    for ifname in ${ifnames[*]}
    do
        result=`ifconfig $ifname 2>&1 | grep -v "error"`
        if [[ x$result != x ]]
        then
            ip=`echo "$result" | grep inet\ | awk '{print $2}' | awk -F: '{print $2}'`
            echo $ip
            return
        fi
    done
    exit -1
}

__find_netip() {
    result=`curl -s icanhazip.com`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    result=`wget -qO - ifconfig.co`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    result=`curl ipecho.net/plain`
    if [[ x$result != x ]]
    then
        echo "$result"
        return
    fi
    exit -1
}

hostname=`hostname`
hostlanip=$(__find_lanip)
hostnetip=$(__find_netip)

echo -e "\n####\thostname($hostname), hostlanip($hostlanip), hostnetip($hostnetip)\t####"

if [[ x$hostnetip == x ]]
then
    echo "Cann't get netip"
    exit -1
fi

raceai_service_name=raceai
raceai_addr=$hostlanip
raceai_port=9119

export HOST_NAME=${hostname}
export HOST_LANIP=${hostlanip}
export HOST_NETIP=${hostnetip}

__start_raceai()
{
    python3 ${top_dir}/services/k12ai_service.py \
        --host ${raceai_addr} \
        --port ${raceai_port}
}

__main()
{
    __start_raceai
}

__main $@
