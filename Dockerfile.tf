FROM tensorflow/tensorflow:2.3.2-gpu

LABEL maintainer="raceai@hzcsai.com"

ARG VENDOR
ARG PROJECT
ARG REPOSITORY
ARG TAG
ARG DATE
ARG VERSION
ARG URL
ARG COMMIT
ARG BRANCH

LABEL org.label-schema.schema-version="1.0" \
      org.label-schema.build-date=$DATE \
      org.label-schema.name=$REPOSITORY \
      org.label-schema.description="HZCS RACE Base" \
      org.label-schema.url="https://www.hzcsai.com" \
      org.label-schema.vcs-url=$URL \
      org.label-schema.vcs-ref=$COMMIT \
      org.label-schema.vcs-branch=$BRANCH \
      org.label-schema.vendor=$VENDOR \
      org.label-schema.version=$VERSION

LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive \
    APT_INSTALL="apt install -y --no-install-recommends" \
    PIP_INSTALL="pip3 install --no-cache-dir --retries 20 --timeout 120 --trusted-host mirrors.intra.didiyun.com --index-url http://mirrors.intra.didiyun.com/pip/simple"

RUN sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.intra\.didiyun\.com\/ubuntu\//g' /etc/apt/sources.list && \
    rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update --fix-missing && $APT_INSTALL \
        cmake locales \
        pkg-config tzdata iputils-ping net-tools \
        bzip2 unzip wget git \
        libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev \
        libv4l-dev libxvidcore-dev libx264-dev libatlas-base-dev gfortran \
        libgl1-mesa-glx ffmpeg \
        python3-dev python3-pip python3-setuptools python3-wheel \
        vim

RUN $PIP_INSTALL -U pip && \
        $PIP_INSTALL scikit-build zmq protobuf pyhocon omegaconf && \
        $PIP_INSTALL opencv-python==4.3.0.36 opencv-contrib-python \
        scikit-learn scipy==1.5.2 matplotlib tqdm==4.48.2 \
        Minio GPUtil

SHELL ["/bin/bash"]
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
