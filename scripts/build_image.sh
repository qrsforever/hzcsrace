#!/bin/bash
#=================================================================
# date: 2020-11-09
# title: build_image
# author: QRS
#=================================================================

export LANG="en_US.utf8"

CUR_FIL=${BASH_SOURCE[0]}
TOP_DIR=`cd $(dirname $CUR_FIL)/..; pwd`

VENDOR=hzcsai_com

MAJOR_RACEAI=1
MINOR_RACEAI=1

DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=$(git describe --tags --always)
URL=$(git config --get remote.origin.url)
COMMIT=$(git rev-parse HEAD | cut -c 1-7)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
NUMBER=$(git rev-list HEAD | wc -l | awk '{print $1}')

echo "DATE: $DATE"
echo "VERSION: $VERSION"
echo "URL: $URL"
echo "COMMIT: $COMMIT"
echo "BRANCH: $BRANCH"
echo "NUMBER: $NUMBER"

__build_image()
{
    PROJECT=$1
    MAJOR=$2
    MINOR=$3

    TAG=$MAJOR.$MINOR.$NUMBER
    REPOSITORY=$VENDOR/$PROJECT

    build_flag=0

    force=$4
    dfile=$5
    if [[ x$force == x0 ]]
    then
        items=($(docker images --filter "label=org.label-schema.name=$REPOSITORY" --format "{{.Tag}}"))
        count=${#items[@]}
        if (( $count == 0 ))
        then
            build_flag=1
        else
            lastest=0
            i=0
            echo "Already exist images:"
            while (( i < $count ))
            do
                echo -e "\t$(expr 1 + $i). $REPOSITORY:${items[$i]}"
                if [[ $lastest != 1 ]] && [[ $(echo ${items[$i]} | cut -d \. -f1-2) == $MAJOR.$MINOR ]]
                then
                    lastest=1
                fi
                (( i = i + 1 ))
            done
            if (( $lastest == 0 ))
            then
                echo -ne "\nBuild new image: $REPOSITORY:$TAG (y/N): "
                read result
                if [[ x$result == xy ]] || [[ x$result == xY ]]
                then
                    build_flag=1
                fi
            fi
        fi
    else
        build_flag=1
    fi
    if (( build_flag == 1 ))
    then
        echo "build image: $REPOSITORY:$TAG"

        docker build --tag $REPOSITORY:$TAG \
            --build-arg VENDOR=$VENDOR \
            --build-arg PROJECT=$PROJECT \
            --build-arg REPOSITORY=$REPOSITORY \
            --build-arg TAG=$TAG \
            --build-arg DATE=$DATE \
            --build-arg VERSION=$VERSION \
            --build-arg URL=$URL \
            --build-arg COMMIT=$COMMIT \
            --build-arg BRANCH=$BRANCH \
            --file $dfile .

        if [[ $? != 0 ]]
        then
            echo "docker build $REPOSITORY:$TAG fail"
            exit $?
        fi
        docker tag $REPOSITORY:$TAG $REPOSITORY
    else
        echo "No need build new image $REPOSITORY!"
    fi
}

# ./build_image.sh Dockerfile.app

__main()
{
    dfile="Dockerfile"
    dname=
    if [[ x$1 != x ]]
    then
        dfile=$1
        dname="_"${dfile#*.}
    fi
    __build_image "raceai${dname}" $MAJOR_RACEAI $MINOR_RACEAI 1 $dfile
}

__main $@
