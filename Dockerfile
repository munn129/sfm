FROM continuumio/anaconda3:latest

RUN apt update
RUN apt install -y libgl1-mesa-glx
RUN conda create -n sfm
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate sfm
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install tqdm