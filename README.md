# ProtoAU

Pytorch implementation of ProtoAU for recommendation.
We present the **Proto**typical contrastive learning through **A**lignment and **U**niformity for recommendation, which is called **ProtoAU**.
A contrastive learning method for recommendation that excels in capturing intricate relationships between user and item interactions, which enhance the basic GNN-based recommendation model's generalization ability and robustness.

Thanks for following our work! :)

## Prepare

There are two environment you can choose: nvidia-docker environment or normal environment.

- For nvidia-docker users, you need to install nvidia-docker2 and restart docker service.

```sh
# docker env
docker build -t protoau .
docker run -itd --gpus all --name protoau
docker exec -it protoau /bin/bash # enter the container
```

- For normal users, you need to install pytorch and other packages. here we use follow environment:
  - Python 3.6
  - Pytorch 1.9 (GPU version)
  - CUDA 11.1
  - cudnn 8

then run follow command to install other packages:

```sh
pip install -r requirements.txt
```

## Quickstart

- Arguments:
  - Config the model arguments in `conf/ProtoAU.yaml`


- Train:

```sh
# train
nohup python index.py --gpu_id=0 --model=ProtoAU --run_name=ProtoAU --dataset=yelp2018 > ./0.log 2>&1 &

# Parallel train(optional)
wandb sweep --project sweep_parallel ./sweep/ProtoAU.yaml # step 1
wandb agent --count 5 oceanlvr/sweep_parallel/[xxx] # replace the [xxx] with your sweep id (step 1 generated)
```
3. For all metric results, you could see the output in the `./0.log` file or the wandb dashboard.
4. For visualizing the results, run python3 `visualize/feature.py`.


## Datasets

<div>
 <table class="table table-hover table-bordered">
  <tr>
    <th rowspan="2" scope="col">DataSet</th>
  <tr>
    <th class="text-center">Users</th>
    <th class="text-center">Items</th>
    <th class="text-center">Ratings</th>
    <th class="text-center">Density</th>
    </tr>   
   <tr>
    <td><a href="https://pan.baidu.com/s/1hrJP6rq" target="_blank"><b>Douban</b></a> </td>
    <td>2,848</td>
    <td>39,586</td>
    <td>894,887</td>
    <td>0.794%</td>
    </tr> 
	 <tr>
    <td><a href="http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip" target="_blank"><b>LastFM</b></a> </td>
    <td>1,892</td>
    <td>17,632</td>
    <td>92,834</td>
    <td>0.27%</td>
    </tr> 
    <tr>
    <td><a href="https://www.dropbox.com/sh/h97ymblxt80txq5/AABfSLXcTu0Beib4r8P5I5sNa?dl=0" target="_blank"><b>Yelp</b></a> </td>
    <td>19,539</td>
    <td>21,266</td>
    <td>450,884</td>
    <td>0.11%</td>
    </tr>
    <tr>
    <td><a href="https://www.dropbox.com/sh/20l0xdjuw0b3lo8/AABBZbRg9hHiN42EHqBSvLpta?dl=0" target="_blank"><b>Amazon-Book</b></a> </td>
    <td>52,463</td>
    <td>91,599</td>
    <td>2,984,108</td>
    <td>0.11%</td>
    </tr>  
  </table>
</div>


## Reference

- https://github.com/coder-Yu/SELFRec/
- https://github.com/RUCAIBox/RecBole2.0


## Cite

Please cite our paper [ieeexplore.ieee.org/document/10650218](https://ieeexplore.ieee.org/document/10650218/) if you use this code.
