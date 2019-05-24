# Instructions for how to run various benchmarks and reproduce numbers in the paper

These instructions are for the devgpu machines inside FAIR.

## VOC dataset
VOC2007 dataset is available on gfsai mount:

```bash
/mnt/fair/VOC2007
```

## Generate the input data files for VOC07
Given the input VOC07 data, for each data partition trainval/test, we want to
generate two files for each partition. These files contain information about the
images and their labels respectively. For example, see the `TRAIN.DATA_FILE`,
`TRAIN.LABELS_FILE`, `TEST.DATA_FILE` and `TEST.LABELS_FILE` in the config file:

```
/experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/full_finetune_voc/caffenet_bvlc_supervised_voc07_finetune_full.yaml
```

To generate these files, follow the instructions [here](../extra_script/README_FB.md).

## Build for local testing

```bash
buck build @mode/dev-nosan -c python.native_link_strategy=separate experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/...
```

## Run locally on devgpu
We use the yaml configuration files to specify the settings for running models.
The yaml configs for the tables in the paper are provided here. These configs
are for running models on 1-gpu machines. If you have different gpus requirement,
you will need to modify learning rate schedule, base lr, number of iterations
etc.


```bash
buck-out/gen/experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/train_net.par --config_file experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/full_finetune_voc/<your_config_name>.yaml
```

## Run on FAIR cluster

Run the following command:

```bash
cd ~/fbsource/fbcode/experimental/deeplearning/prigoyal/self_supervision_benchmark && GPU_TYPE=P100 CFG=configs/legacy_tasks/full_finetune_voc/<your_config>.yaml NAME=caffenet_bvlc_in1k_supervised_in1k_tune REBUILD=true ./deploy/fb/launch.sh
```
