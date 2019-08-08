## Buck build

```
buck build @mode/dev-nosan -c python.native_link_strategy=separate experimental/deeplearning/yihuihe/self_supervision_benchmark/tools/svm/...
```

## Low-shot test svm

```
buck-out/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/tools/svm/test_svm_low_shot.py --data_file /mnt/vol/gfsai-bistro-east/ai-group/users/prigoyal/caffe2/resnet/gen/oss_benchmark/voc2007/oss_rn50_jigsaw_294init/test_res_conv1_bn_s0_resize_features.npy --targets_data_file /mnt/vol/gfsai-bistro-east/ai-group/users/prigoyal/caffe2/resnet/gen/oss_benchmark/voc2007/oss_rn50_jigsaw_294init/test_res_conv1_bn_s0_resize_targets.npy --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" --output_path /mnt/vol/gfsai-bistro-east/ai-group/users/prigoyal/caffe2/resnet/gen/oss_benchmark/voc2007/oss_rn50_jigsaw_294init/low_shot/svm_conv1/ --k_values "1,2,4,8,16,32,64,96" --sample_inds "0,1,2,3,4"
```

## Aggregate stats
```
buck-out/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/tools/svm/aggregate_low_shot_svm_stats.py --output_path /mnt/vol/gfsai-bistro-east/ai-group/users/prigoyal/caffe2/resnet/gen/oss_benchmark/voc2007/oss_rn50_jigsaw_294init/low_shot/svm_conv1/ --k_values "1,2,4,8,16,32,64,96" --sample_inds "0,1,2,3,4"
```
