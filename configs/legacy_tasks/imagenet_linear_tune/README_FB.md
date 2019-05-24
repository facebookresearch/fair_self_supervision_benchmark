# Instructions for how to run various benchmarks and reproduce numbers in the paper

These instructions are for the devgpu machines inside FAIR.

## Installing ImageNet-1K dataset on devgpu
First, check whether the following path on your devgpu exists or not:

```bash
ls /data/local/packages/ai-group.imagenet-full-size/prod/imagenet_full_size/
```

If the path exist, you can skip this section. If the path does not exist, you
can install the ImageNet-1K dataset on your devgpu by following commands:
Below are some common instructions used

0. Edit `/data/local/packages/packages.txt` and add `ai-group.imagenet-full-size`.
If the file does NOT exist, you can create it.

0. Run `sudo chefctl -i` and wait for the chef run to finish. After the chef run,
your data will be under `/data/local/packages/ai-group.imagenet-full-size`. NOTE
that the chef run might take some time to finish. Please be patient with it while
it runs.

## Generate the input data files for ImageNet
Given the input ImageNet data, for each data partition train/val, we want to
generate two files for each partition. These files contain information about the
images and their labels respectively. For example, see the `TRAIN.DATA_FILE`,
`TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` in the config file:

```
/experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/imagenet_linear_tune/caffenet_bvlc_supervised_finetune_linear.yaml
```

To generate these files, follow the instructions [here](../extra_script/README_FB.md).

## Build for local testing

```bash
buck build @mode/dev-nosan -c python.native_link_strategy=separate experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/...
```

## Run locally on devgpu
We use the yaml configuration files to specify the settings for running models.
The yaml configs for the tables in the paper are provided here. These configs
are for running models on 8-gpu machines. If you have different gpus requirement,
you will need to modify learning rate schedule, base lr, number of iterations
etc.


```bash
buck-out/gen/experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/train_net.par --config_file experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/imagenet_linear_tune/<your_config_name>.yaml
```

## Run on FAIR cluster

Run the following command:

```bash
cd ~/fbsource/fbcode/experimental/deeplearning/prigoyal/self_supervision_benchmark && GPU_TYPE=P100 CFG=configs/legacy_tasks/imagenet_linear_tune/<your_config>.yaml NAME=caffenet_bvlc_in1k_supervised_in1k_tune REBUILD=true ./deploy/fb/launch.sh
```
