set -e
set -x

#action can be extract svm svmtest
action=${ACTION-extract}
list=${LIST-extract_list.txt}
iterlist=${ITERLIST-exp/shortiter.txt}
if [[ $list != *"conv5"* ]]; then
  # layers=('res4_5_branch2c_bn_s0_s3k8_resize' 'res3_3_branch2c_bn_s0_s5k13_resize' 'res2_2_branch2c_bn_s0_s8k16_resize' 'res_conv1_bn_s0_resize')
  # layers=('res3_3_branch2c_bn_s0_s5k13_resize' 'res2_2_branch2c_bn_s0_s8k16_resize' 'res_conv1_bn_s0_resize')
  layers=('res4_5_branch2c_bn_s0_s3k8_resize')
else
  # layers=('res5_2_branch2c_bn_s0_s1k6_resize' 'res4_5_branch2c_bn_s0_s3k8_resize' 'res3_3_branch2c_bn_s0_s5k13_resize' 'res2_2_branch2c_bn_s0_s8k16_resize' 'res_conv1_bn_s0_resize')
  # layers=('res3_3_branch2c_bn_s0_s5k13_resize' 'res2_2_branch2c_bn_s0_s8k16_resize' 'res_conv1_bn_s0_resize')
  layers=('res5_2_branch2c_bn_s0_s1k6_resize' 'res4_5_branch2c_bn_s0_s3k8_resize')
  # layers=('res5_2_branch2c_bn_s0_s1k6_resize')
fi

root=${ROOT-/home/yihuihe/caffe2/resnet/gen/voc2007_new}
for line in $(cat $list) ; do
  # for iter in c2_model_iter5005 c2_model_iter10010 bestModel latest ; do
  for iter in $(cat $iterlist) ; do # c2_model_iter50050 c2_model_iter45045
    p=$line/imagenet1k/resnet_midlevel_delayed_fc/depth_50/${iter}.pkl
    echo extracting $p
    for layer in "${layers[@]}" ; do
    # for layer in res4_5_branch2c_bn_s0_s3k8_resize ; do
      if [[ $action == extract ]] ; then
        ROOT=$root PARAM=$p ACTION=$action LAYER=$layer bash extract.sh & #&& sleep 20s
      else
        ROOT=$root PARAM=$p ACTION=$action LAYER=$layer bash extract.sh &
      fi
      # if [[ -e $p ]] ; then
      #   if [[ $action == extract ]] ; then
      #     PARAM=$p ACTION=$action LAYER=$layer bash extract.sh #&& sleep 5s
      #   else
      #     PARAM=$p ACTION=$action LAYER=$layer bash extract.sh &
      #   fi
      # else
      #   echo notexist
      #   exit 1
      # fi
    done
  done
done

wait
echo "All done"
