# Submission 549
We provide the source code for reproducing Figure 9b in the main text. A pre-trained network is employed.

## Prerequisites
We assume a fresh install of Ubuntu 20.04. For example,

```
docker run --gpus all --shm-size 128G -it --rm -v $HOME:/home/ubuntu ubuntu:20.04
```

Install python and pip:
```
apt-get update
apt install python3-pip
```

## Dependencies
Install python package dependencies through pip:

```
pip install -r requirements.txt
```

## Execute
```
python3 online/execute_online.py -device [device]
```
[device] is either cpu or cuda

Solved results are stored in the output directory.
