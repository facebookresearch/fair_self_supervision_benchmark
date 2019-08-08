# cp -r /mnt/homedir/yihuihe/outs/*imgs.npy /home/yihuihe/outs/
a=cluster
PARAM=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/latest.pkl ACTION=$a bash extractclustervis.sh
PARAM=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/5005.pkl ACTION=$a bash extractclustervis.sh
PARAM=/home/yihuihe/20190712/sigma5.0_top8_LR0.01_FC16_configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale_conv5_pca_cin1k.yaml.07_44_38.tAjEMIgJ/imagenet1k/resnet_midlevel_delayed_fc/depth_50/10010.pkl ACTION=$a bash extractclustervis.sh
