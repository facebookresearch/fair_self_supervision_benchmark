set -x
DIR=${DIR-/tmp/ssl-benchmark-output/extract_features/weights_init}
SVM=${SVM-/tmp/ssl-benchmark-output/voc07_svm/svm_conv45/}
LAYER=${LAYER-res4_5_branch2c_bn_s0_s3k8_resize}
PARAM=${PARAM-/data/users/yihuihe/scratch/20181030/rn50_jigsaw_10kperm_yfcc100m_10ep_configs/unsupervised/jigsaw/pretext/resnet/in1k_schedule/yfcc100m_siamese_BN_bias_decay_shift_scale.yaml.06_30_10.90Ykc6Kn/yfcc100m/resnet_jigsaw_siamese_2fc/depth_50/c2_model_iter3903900.pkl}

mkdir -p $DIR
rm -r $DIR
mkdir -p $DIR

mkdir -p $SVM
rm -r $SVM
mkdir -p $SVM

set -e
/home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml --output_dir $DIR TEST.PARAMS_FILE $PARAM
/home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml --output_dir $DIR --data_type test --output_file_prefix test TEST.PARAMS_FILE $PARAM
/home/yihuihe/self_supervision_benchmark/tools/svm/train_svm_kfold.py --costs_list 0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0 --data_file $DIR/trainval_${LAYER}_features.npy --targets_data_file $DIR/trainval_${LAYER}_targets.npy --output_path $SVM
/home/yihuihe/self_supervision_benchmark/tools/svm/test_svm.py --costs_list 0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0 --data_file $DIR/test_${LAYER}_features.npy --targets_data_file $DIR/test_${LAYER}_targets.npy --output_path $SVM
