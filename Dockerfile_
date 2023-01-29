FROM ufoym/deepo:1.8.0.dev20210103

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

# Base

RUN APT_INSTALL="apt install -y --no-install-recommends" && \
    sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.intra\.didiyun\.com\/ubuntu\//g' /etc/apt/sources.list && \
    apt update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential cmake unzip pkg-config tzdata iputils-ping net-tools \
        libjpeg-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libatlas-base-dev gfortran \
        jq libgl1-mesa-glx ffmpeg

RUN PIP_INSTALL="pip install --no-cache-dir --retries 20 --timeout 120 \
        --trusted-host mirrors.intra.didiyun.com \
        --index-url http://mirrors.intra.didiyun.com/pip/simple" && \
    $PIP_INSTALL pip && pip uninstall -y enum34 && \
    $PIP_INSTALL GPUtil psutil packaging zerorpc flask flask_cors omegaconf \
    onnx onnxruntime yacs ffmpeg-python face_alignment \
    torchsummary tensorboard seaborn \
    pyhocon protobuf redis "jsonnet>=0.10.0" && \
    pip uninstall -y opencv-python

# install opencv (dobble)

# COPY external/opencv/4.3.0.tar.gz 4.3.0.tar.gz
# RUN tar zxf 4.3.0.tar.gz && cd opencv-4.3.0 && mkdir build && cd build && \
#     cmake -D CMAKE_BUILD_TYPE=RELEASE \
#         -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
#         -D ENABLE_FAST_MATH=ON \
#         -D INSTALL_PYTHON_EXAMPLES=OFF \
#         -D INSTALL_C_EXAMPLES=OFF \
#         -D OPENCV_ENABLE_NONFREE=OFF \
#         -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
#         -D PYTHON_EXECUTABLE=$(which python) \
#         -D PYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#         -D PYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
#         -D BUILD_EXAMPLES=OFF \
#         -D BUILD_JPEG=ON .. && \
#     make -j8 && make install && ldconfig && cd ../.. && rm -rf 4.3.0.tar.gz opencv-4.3.0

# update python3

# RUN update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.8 1 && \
#     update-alternatives --install /usr/local/bin/python  python  /usr/bin/python3.8 1

# pytorch-lightning

# COPY projects/pytorch-lightning /raceai/codes/projects/pytorch-lightning
# RUN cd /raceai/codes/projects/pytorch-lightning && \
#     pip install --no-cache-dir --retries 20 --timeout 120 \
#         --trusted-host mirrors.intra.didiyun.com \
#         --index-url http://mirrors.intra.didiyun.com/pip/simple \
#         --editable .
RUN pip install --no-cache-dir --retries 20 --timeout 120 \
         --trusted-host mirrors.intra.didiyun.com \
         --index-url http://mirrors.intra.didiyun.com/pip/simple \
         pytorch-lightning

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

# Other

RUN pip install --no-cache-dir --retries 20 --timeout 120 \
         --trusted-host mirrors.intra.didiyun.com \
         --index-url http://mirrors.intra.didiyun.com/pip/simple \
         opencv-python
         

SHELL ["/bin/bash"]               
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
