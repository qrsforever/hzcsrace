FROM hzcsk8s.io/pytorch1.12.1_cuda11.3

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

ENV PATH=/raceai/codes/bin:$PATH

RUN $PIP_INSTALL zerorpc flask flask_cors omegaconf redis "jsonnet>=0.10.0" \
        "pytorch-lightning==1.9.0" matplotlib pandas seaborn

SHELL ["/bin/bash"]               
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
