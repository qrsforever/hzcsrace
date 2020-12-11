FROM ufoym/deepo:1.7.0.dev20200819

LABEL maintainer="hzcskrace@hzcsai.com"

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

WORKDIR /raceai

ENV TZ=Asia/Shanghai \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PATH=/raceai/codes/bin:$PATH \
    PYTHONPATH=/raceai/codes/app:$PYTHONPATH    

# Base

RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    PIP_INSTALL="pip install -U --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host mirrors.intra.didiyun.com \
        --index-url http://mirrors.intra.didiyun.com/pip/simple" && \
    sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.intra\.didiyun\.com\/ubuntu\//g' /etc/apt/sources.list && \
    apt update --fix-missing && $PIP_INSTALL install -U pip && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        tzdata iputils-ping jq net-tools libgl1-mesa-glx && \
    pip uninstall enum34 -y && \
    $PIP_INSTALL GPUtil psutil packaging zerorpc \
        flask flask_cors omegaconf opencv-python opencv-contrib-python \
        onnx onnxruntime \
        torchsummary tensorboard seaborn \
        pyhocon protobuf "jsonnet>=0.10.0"

# update python3

# RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.8 1 && \
#     update-alternatives --install /usr/local/bin/python  python  /usr/bin/python3.8 1

# pytorch-lightning

COPY projects/pytorch-lightning /raceai/codes/projects/pytorch-lightning
RUN cd /raceai/codes/projects/pytorch-lightning && \
    pip install --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host mirrors.intra.didiyun.com \
        --index-url http://mirrors.intra.didiyun.com/pip/simple \
        --editable .

# detectron2

COPY projects/detectron2 /raceai/codes/projects/detectron2
RUN cd /raceai/codes/projects/detectron2 && \
    pip install --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host mirrors.intra.didiyun.com \
        --index-url http://mirrors.intra.didiyun.com/pip/simple \
        --editable .

# yolov5
COPY projects/yolov5 /raceai/codes/projects/yolov5
ENV PYTHONPATH=/raceai/codes/projects/yolov5:$PYTHONPATH    

# allennlp


# Clean

# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
#     ldconfig && \
#     apt-get clean && \
#     apt-get autoremove && \
#     rm -rf /var/lib/apt/lists/* /tmp/* ~/*
