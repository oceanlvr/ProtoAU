# ProtoAU

## Prepare

For nvidia-docker users, you need to install nvidia-docker2 and restart docker service.

```sh
# docker env
docker build -t protoau .
docker run -itd --gpus all --name protoau
docker exec -it protoau /bin/bash
```

For normal users, you need to install pytorch and other packages.

here we use follow environment:

- pytorch 1.9 (GPU version)
- CUDA 11.1
- cudnn 8

then run follow command to install other packages:

```sh
pip install -r requirements.txt
```

## Usage

Train
```
nohup python index.py --gpu_id=0 --model=ProtoAU --run_name=ProtoAU --dataset=yelp2018 > ./0.log 2>&1 &
```

## Reference

- https://github.com/coder-Yu/SELFRec/
- https://github.com/RUCAIBox/RecBole2.0


