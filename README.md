# Memorization-Curvature (ICML 2024 Spotlight)

This code is the official implementation for the paper, 
**Memorization Through the Lens of Curvature of Loss Function Around Samples**. This code contains the experiments for the three links described in the paper.

Accepted at ICML 2024 Spotlight (3.5% Acceptance), if you use this github repo consider citing our work
```bibtex
@inproceedings{
    garg2024memorization,
    title={Memorization Through the Lens of Curvature of Loss Function Around Samples},
    author={Garg, Isha and Ravikumar, Deepak and Roy, Kaushik},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=WQbDS9RydY}

}
```
Full paper available [paper link](https://openreview.net/forum?id=WQbDS9RydY).
You may also be interested in our work (ICML '24) on Privacy Memorization and Curvature [paper](https://openreview.net/forum?id=4dxR7awO5n), [code](https://github.com/DeepakTatachar/Privacy-Memorization-Curvature).

## Setup
Create a folder ```./pretrained/<dataset name>``` and ```./pretrained/<dataset name>/temp```
i.e. 
```
mkdir pretrained
mkdir pretrained/imagenet
mkdir pretrained/imagenet/temp
```

Update the paths for the datasets in  ```./utils/load_dataset.py```, replace the string 'Set path'

## Code Flow
1. Train model using ```train_<dataset>.py```
    * Trains the models and saves checkpoint every epoch for analysis in step 2.
2. Run ```score_<dataset>_checkpoint.py```
    * Computes the input loss curvature for the model for each epoch and saves in ```./curv_scores```
3. Use the ```analyze_<dataset>_curv.ipynb``` for analysis

## Training
To train ImageNet model use ```train_imagenet.py```. This code is made to be trained in a distributed manner and makes use of torch run.
To train on 1 node and 1 GPU use the following command
```
OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=<ip-address-of-machine>:26903 \ 
train_imagenet.py
```
For multi-gpu single node use
```
OMP_NUM_THREADS=12 torchrun \
--nproc_per_node=3 \
--nnodes=1 \
--node_rank=0 \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=<ip-address-of-machine>:26903 \ 
train_imagenet.py
```
This uses 3 GPUs on a single node
Output of each node is logged into ```./logs``` folder with a corresponding log file.

## Visualizations
See [visualizations](./images/readme.md) for additional results and higher quality images from the paper.
