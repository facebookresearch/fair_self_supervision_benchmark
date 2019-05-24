# Instructions for how to run various benchmarks and reproduce numbers in the paper

These instructions are for the devgpu machines inside FAIR.

## VOC dataset
VOC2007 dataset is available on gfsai mount:

```bash
/mnt/fair/VOC2007
```

## Build Detectron for local testing

```bash
cd ~/fbsource/fbcode/ && buck build @mode/dev-nosan -c python.native_link_strategy=separate //vision/detection/fair/detectron/tools/...
```

## Run locally on devgpu
We use the yaml configuration files to specify the settings for running models.
The yaml configs for the tables in the paper are provided here. These configs
are for running models on 2-gpu machines. If you have different gpus requirement,
you will need to modify learning rate schedule, base lr, number of iterations
etc.

```bash
buck-out/gen/vision/detection/fair/detectron/tools/train_net.par --cfg experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/object_detection_frozen/voc07/<your_config>.yaml --multi-gpu-testing
```

## Run on FAIR cluster

Run the following command:

```bash
cd ~/fbsource/fbcode/vision/detection/fair/detectron

GPU=2 CPU=6 MEM=24 ./fb/cluster/launch.sh ~/fbsource/fbcode/experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/legacy_tasks/object_detection_frozen/voc07/<your_config>.yaml exp_name [nobuild]
```
