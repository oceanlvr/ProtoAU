# docker build -t protoau .
# docker run -itd --gpus all --name protoau

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
RUN mkdir -p /workspace
WORKDIR /workspace
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
