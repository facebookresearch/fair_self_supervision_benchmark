# Follow the steps below to run the binary on the cluster and locally

###############################################################################
## Run on devgpu
```bash
cd ~/fbsource/fbcode

buck build @mode/dev-nosan \
    -c python.native_link_strategy=separate \
    experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/...

buck-out/gen/experimental/deeplearning/prigoyal/self_supervision_benchmark/tools/train_net.par \
    --config_file experimental/deeplearning/prigoyal/self_supervision_benchmark/configs/<your_config_name>.yaml
```

################################################################################
## Run on cluster
```bash
cd ~/fbsource/fbcode/experimental/deeplearning/prigoyal/self_supervision_benchmark

buck build @mode/opt \
    -c python.native_link_strategy=separate \
    //experimental/deeplearning/prigoyal/self_supervision_benchmark/tools:train_net

GPU_TYPE=P100 CFG=configs/legacy_tasks/imagenet_linear_tune/<your_config>.yaml \
    NAME=caffenet_bvlc_in1k_supervised_in1k_tune REBUILD=true ./deploy/fb/launch.sh
```
