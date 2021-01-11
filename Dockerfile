# A Dockerfile that sets up a full Gym install
FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3.7 \
    python3-pip \
    python3-tk \
    libjpeg8-dev \ 
    libtiff5-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libopenmpi-dev  \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    wget \
    unzip \
    libboost-all-dev \
    g++ \
    make \
    libmpg123-dev \
    libsndfile1-dev \
    chrpath \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install setuptools

WORKDIR /root
ADD setup.py /root/kitt/setup.py
ADD kitt /root/kitt/kitt

WORKDIR /root/kitt
RUN python3.7 -m pip install -e .[all]

ADD utilities /root/kitt/utilities
ADD examples /root/kitt/examples
ADD thesis-experiments /root/kitt/thesis-experiments
