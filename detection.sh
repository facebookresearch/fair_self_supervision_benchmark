param=${PARAM}
build=${BUILD-nobuild}
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

name=det${iter}_${config}
set -e
converted=$(dirname $param)/converted_iter${iter}.pkl
sudo chmod 777 $(dirname $param)
bash convert.sh $param $converted
$HOME/fbsource/fbcode/vision/detection/fair/detectron/fb/cluster/launch.sh configs/benchmark_tasks/object_detection_frozen/voc07/fast_rcnn_R-50-C4_with_ss_proposals_trainval.yaml $name $build TRAIN.WEIGHTS $converted
