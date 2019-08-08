# /home/prigoyal/20181015/jigsaw_rn50_in1k_5k_perms_configs/unsupervised/jigsaw/pretext/resnet/IN1k_8gpu_siamese_BN_bias_decay_shift_scale.yaml.08_29_01.MkJ6IRPy/imagenet1k/resnet_jigsaw_siamese_2fc/depth_50/c2_model_iter450450.pkl
set -x
set -e
launcher=${LAUNCHER-detection.sh}
list=${LIST-exp/det.txt}
iterlist=${ITERLIST-exp/longeriter.txt}
for line in $(cat $list) ; do #exp/det.txt
  for iter in $(cat $iterlist) ; do
    p=$line/imagenet1k/resnet_midlevel_delayed_fc/depth_50/${iter}.pkl
    echo converting $p
    PARAM=$p BUILD=nobuild bash $launcher &
  done
done
wait
echo all done
