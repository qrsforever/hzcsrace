FROM hzcsk8s.io/pytorch1.12.1_cuda11.3

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
    PYTHONPATH=/raceai/codes/app:/raceai/codes/projects:$PYTHONPATH    

# pytorch-lightning

RUN $PIP_INSTALL pytorch-lightning

# detectron2

COPY projects/detectron2 /raceai/codes/projects/detectron2
RUN cd /raceai/codes/projects/detectron2 && $PIP_INSTALL --editable .

# yolov5
COPY projects/yolov5 /raceai/codes/projects/yolov5
ENV PYTHONPATH=/raceai/codes/projects/yolov5:$PYTHONPATH    

# Clean
RUN apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

SHELL ["/bin/bash"]               
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
