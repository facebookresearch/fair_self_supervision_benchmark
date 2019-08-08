
param=${PARAM}
CFG=${CFG-configs/benchmark_tasks/image_classification/voc07/resnet50_jigsaw_extract_features.yaml}
#action can be extract svm svmtest
action=${ACTION-extract}
if [[ $param != *"bistro/gpu"* ]]; then
  if [[ -e /home/$param ]]; then
    param=/home/$param
  elif [[ -e /home/$param ]]; then
    param=/home/$param
  else
    echo nonexist
    exit 0
  fi
fi

set -x

get_cfgs(){
  IN=$1
  arr=(${IN//// })
  cfg=${arr[8]}
  cfg=${cfg//_configs/ }
  iter=${arr[17]}
  iter=${iter//c2_model_iter/ }
  iter=${iter//\.pkl/ }
  yaml=${arr[13]}
  yaml=(${yaml//\.yaml/ })
  yaml=${yaml[0]}
  echo $cfg $iter $yaml
}

read config iter yaml < <(get_cfgs $param)

ROOT=${ROOT-/home/yihuihe/caffe2/resnet/gen/voc2007_new}
root=$ROOT/$yaml
path=svm${iter}_${config}
layer=${LAYER-res4_5_branch2c_bn_s0_s3k8_resize}
outpath=$root/$path
name=${NAME-$action$path$layer}
set -e
mkdir -p -m 777 ${outpath}
if [[ $action == extract ]] ; then
  for i in 'train' 'test' ; do
      NAME=$name \
      TYPE=$i \
      OUTPUT_PREFIX=$i \
      OUTPUT_DIR=${outpath} \
      MODEL=$param \
      CFG=$CFG \
      ./deploy/extract_features.sh
  done
elif [[ $action == svm ]] ; then
  NAME=$name \
  data_file=${outpath}/train_${layer}_features.npy \
  targets_data_file=${outpath}/train_${layer}_targets.npy \
  costs_list=0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0 \
  output_path=${outpath}/${layer} \
  ./deploy/train_svm.sh
elif [[ $action == svmtest ]] ; then
  NAME=${name} \
  data_file=${outpath}/test_${layer}_features.npy \
  targets_data_file=${outpath}/test_${layer}_targets.npy \
  costs_list="0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
  output_path=${outpath}/${layer} ./deploy/test_svm.sh
  # output_path=${outpath} ./deploy/test_svm.sh
  # $CODE/buck-out/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/tools/svm/test_svm.py \
  # --data_file ${outpath}/test_${layer}_features.npy \
  # --targets_data_file ${outpath}/test_${layer}_targets.npy \
  # --costs_list "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0" \
  # --output_path ${outpath}
elif [[ $action == vis ]] ; then
  $CODE/buck-out/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/tools/extract_features.py \
  --action visap \
  --output_dir $root
fi
