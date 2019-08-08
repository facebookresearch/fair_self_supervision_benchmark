set -e
set -x
# out=/home/yihuihe/conv5_in1k_init_bnfreeze_pca_sigma5.0_top8_LR0.01_FC16_outs/
# for model in /home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/latest.pkl \
# /home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter5005.pkl \
# /home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter10010.pkl ; do
#   PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=npyextract PLACE=local bash extractclustervis.sh
#   PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=cluster PLACE=local bash extractclustervis.sh
#   PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=viscluster PLACE=local bash extractclustervis.sh
# done

# model=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/latest.pkl
# model=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter0.pkl
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=npyextract PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=cluster PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=viscluster PLACE=local bash extractclustervis.sh
# model=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter5005.pkl
# # PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=npyextract PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=cluster PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=viscluster PLACE=local bash extractclustervis.sh
# model=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter10010.pkl
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=npyextract PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=cluster PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=viscluster PLACE=local bash extractclustervis.sh
# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ REFERENCE_DIR=/home/yihuihe/visouts/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k/sigma5.0_top8_LR0.01_FC16/0/ ACTION=comparecluster PLACE=local bash extractclustervis.sh

# PARAM=$model ROOTDIR=/home/yihuihe/visouts/ REFERENCE_DIR=/home/yihuihe/visouts/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k/sigma5.0_top8_LR0.01_FC16/5005/ ACTION=comparecluster PLACE=local bash extractclustervis.sh

# /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CODE/experimental/deeplearning/yihuihe/self_supervision_benchmark/configs/benchmark_tasks/image_classification/in1k/resnet_extract_features.yaml --output_dir /home/yihuihe/in1k_outs+$out --data_type test --action comparecluster EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $model


# extract second epoch
out=/home/yihuihe/conv4_yfcc_scratch_pca_sigma5.0_top2_LR0.01_FC16_outs/
for model in /home/yihuihe/20190712/sigma5.0_top2_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv4_scratch_pca.yaml.07_38_06.stT8d46k/imagenet1k/resnet_midlevel_delayed_fc/depth_50/latest.pkl \
/home/yihuihe/20190712/sigma5.0_top2_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv4_scratch_pca.yaml.07_38_06.stT8d46k/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter10010.pkl \
/home/yihuihe/20190712/sigma5.0_top2_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv4_scratch_pca.yaml.07_38_06.stT8d46k/imagenet1k/resnet_midlevel_delayed_fc/depth_50/c2_model_iter5005.pkl ; do
  # PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=npyextract PLACE=local bash extractclustervis.sh EXTRACT.SAVE_IMGS False
  # PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=cluster PLACE=local bash extractclustervis.sh
  PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=assign PLACE=local bash extractclustervis.sh
  PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=viscluster PLACE=local bash extractclustervis.sh
  # PARAM=$model ROOTDIR=/home/yihuihe/visouts/ ACTION=comparecluster PLACE=local bash extractclustervis.sh
done
