param=${PARAM}
build=${BUILD-nobuild}
#action can be extract svm svmtest
# action=${ACTION-extract}
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

set -e
# name=places${iter}_${config}
#TAG=places NAME=$name CFG=configs/benchmark_tasks/image_classification/places205/resnet50_jigsaw_finetune_linear.yaml ./deploy/launch.sh TRAIN.PARAMS_FILE $param MIDLEVEL.RECLUSTER_PERIOD 50000000
name=imagenet${iter}_${config}
TAG=imagenet NAME=$name CFG=configs/legacy_tasks/imagenet_linear_tune/resnet50_jigsaw_finetune_linear.yaml ./deploy/launch.sh TRAIN.PARAMS_FILE $param MIDLEVEL.RECLUSTER_PERIOD 50000000
