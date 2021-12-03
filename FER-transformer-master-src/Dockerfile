from nvcr.io/nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04
#For base image see https://ngc.nvidia.com/catalog/containers?orderBy=modifiedDESC&pageNumber=1&query=&quickFilter=&filters=
#It allows you to have an image with cuda and cudnn already installed. Many version exists on the website.
#There is also images with preinstalled frameworks (e.g. pytorch, tensorflow), but I decided to not use them here.


#Avoid interactive mode that ask geographic area for the docker build
ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && \

apt-get install -y python3.7 python3-pip python3-tk imagemagick
#apt-get install python3-matplotlib

#For opencv-python : apt-get install -y libsm6 libxext6 libxrender-dev (https://github.com/NVIDIA/nvidia-docker/issues/864)
#For matplotlib : apt-get install -y python3-tk imagemagick

ADD requirements.txt /
RUN cd / && \
	mkdir python && \
	pip3 install -r requirements.txt
	
