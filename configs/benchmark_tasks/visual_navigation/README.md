# Benchmark Task: Visual Navigation

This document provides instructions on how to test the feature representation quality of models on Visual Navigation using Gibson. We use a docker image for this purpose, graciously provided by Sax et al. corresponding to their [work](https://arxiv.org/pdf/1812.11971.pdf).

This repository does not contain any code relevant to the Visual Navigation training. Instead, the docker image has the dataset and the relevant code.

**Requirements:**
- We use EC2 instance with 8GPU, 32CPU, 200GB SSD, Ubuntu 16.04
- NVIDIA GPU

Follow the instructions below:

#### Step 1: Download the docker
```bash
docker pull activeperception/midlevel-training:0.3
```

#### Step 2: Setup the Visdom server

```bash
pip install visdom
pip install -m visdom.server
# Run the command curl https://ipinfo.io/ip to get the Public IP of your machine and view the visom server at publicIP:8087
```

#### Step 3: Visual Navigation training

- We first run the docker image using the following instructions:

```bash
# on your machine
export DISPLAY=:0.0
xhost +local:root
docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp/:/tmp/ activeperception/midlevel-training:0.3
# inside docker now
cd ../teas/notebooks/
```

- Now, select the model weights you want to use. The weights are provided in [`MODEL_ZOO`](../../../MODEL_ZOO.md). We have converted the original caffe2 models to PyTorch model using the scripts provided [here](../../../extra_scripts/README.md). For demonstration purpose, we will use the Jigsaw model pre-trained on ImageNet-22k. Download the model to your local disk:

```bash
cd /tmp && wget http://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/models/resnet50_jigsaw_in22k_pretext_gibson.dat
```

- The ResNet-50 model in the docker image has a compression Conv layer to compress the final layer representation from `2048 x 16 x 16 -> 8 x 16 x 16`. This compression layer is not part of the standard ResNet-50 model and hence we do NOT have weights initialization for this. We keep this layer randomly initialized. Hence we make a small code modification to the docker image code so that our models can load.

```bash
# apply the changes highlighted in the gist https://gist.github.com/prigoyal/c48d805b8edafb3ffb87344ceba8da34#file-architectures-py-L454-L455
vim ../teas/models/architectures.py
```

- Next, we make some trivial code changes corresponding to features from various layers that we want to train on. In our [paper, Figure 6](https://arxiv.org/pdf/1905.01235.pdf), these layers are `res3`, `res4` and `res5`.

```bash
# apply the following changes for res3, res4 or res5 as desired
# res3: https://gist.github.com/prigoyal/057a4e3aec30c05446c6970b13f4ec49#file-taskonomyencoder-py-L23-L27 and https://gist.github.com/prigoyal/057a4e3aec30c05446c6970b13f4ec49#file-taskonomyencoder-py-L72-L76
#
# res4: https://gist.github.com/prigoyal/fdce7612d3a314ae9d51e5e1489f77c6#file-taskonomyencoder-py-L24-L27 and https://gist.github.com/prigoyal/fdce7612d3a314ae9d51e5e1489f77c6#file-taskonomyencoder-py-L73-L76
#
# res5:  https://gist.github.com/prigoyal/5bb8050f8c53064e6ff65322e7cf30f7#file-taskonomyencoder-py-L26 and https://gist.github.com/prigoyal/5bb8050f8c53064e6ff65322e7cf30f7#file-taskonomyencoder-py-L75
vim ../teas/models/taskonomyencoder.py
```

- Now, we are ready to train. Following are the example command for training using the model weights and training for random initialization. We do 5 independent training runs using the same setup and report the mean/std results as in [Figure 6 of our paper](https://arxiv.org/pdf/1905.01235.pdf).

    - For random initialization:

        ```bash
        python Gibson_baseline.py \
            /tmp/random_run1 run_cfg with uuid=aws_navigate_random_run1 \
            cfg.saving.visdom_server='<your_server>' cfg_navigation \
            cfg.env.start_locations_file=None \
            cfg.training.num_frames=400000
        ```
    - For initalization using weights from supervised/self-supervised training:

        ```bash
        python Gibson_baseline.py \
            /tmp/in22k_run1 run_cfg with uuid=aws_navigate_in22k_run1 \
            cfg.saving.visdom_server='<your_server>' cfg_navigation \
            cfg.env.start_locations_file=None \
            cfg.learner.perception_network='features_only' \
            cfg.learner.taskonomy_encoder='/tmp/resnet50_jigsaw_in22k_pretext_gibson.dat' \
            cfg.training.num_frames=400000
        ```
