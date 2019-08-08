set -x

param=${PARAM}
CFG=${CFG-configs/benchmark_tasks/image_classification/in1k/resnet_extract_features_npy.yaml}
ROOTDIR=${ROOTDIR-/home/yihuihe/visouts/}
REFERENCE_DIR=${REFERENCE_DIR-/home/yihuihe/outs}
#action can be extract svm svmtest
action=${ACTION-npyextract}
place=${PLACE-cluster}

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

root=$ROOTDIR/${yaml}/${config}/${iter}
name=${NAME-$action${yaml}/${config}/${iter}}
set -e
mkdir -p -m 777 ${root}
if [[ $action == npyextract ]] ; then
  if [[ $place == cluster ]] ; then
    NAME=$name \
    TYPE=test \
    OUTPUT_PREFIX=trainval \
    OUTPUT_DIR=${root} \
    MODEL=$param \
    CFG=$CFG \
    ./deploy/clustervis.sh --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True "$@"
  elif [[ $place == local ]] ; then
    /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CFG --output_dir $root --data_type test --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $param EXTRACT.IMGS_DIR /home/yihuihe/outs NUM_DEVICES 1 TRAIN.BATCH_SIZE 200 TEST.BATCH_SIZE 200 "$@"
  fi
elif [[ $action == cluster ]] ; then
  if [[ $place == cluster ]] ; then
    NAME=$name \
    TYPE=test \
    OUTPUT_PREFIX=trainval \
    OUTPUT_DIR=${root} \
    MODEL=$param \
    CFG=$CFG \
    ./deploy/clustervis.sh --action $action "$@"
  elif [[ $place == local ]] ; then
    /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CFG --output_dir $root --data_type test --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $param EXTRACT.IMGS_DIR /home/yihuihe/outs NUM_DEVICES 1 TRAIN.BATCH_SIZE 200 TEST.BATCH_SIZE 200 "$@"
  fi
elif [[ $action == viscluster ]] ; then
  if [[ $place == cluster ]] ; then
    NAME=$name \
    TYPE=test \
    OUTPUT_PREFIX=trainval \
    OUTPUT_DIR=${root} \
    MODEL=$param \
    CFG=$CFG \
    ./deploy/clustervis.sh --action $action --reference_dir /home/yihuihe/outs EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True "$@"
  elif [[ $place == local ]] ; then
    /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CFG --output_dir $root --reference_dir $REFERENCE_DIR --data_type test --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $param EXTRACT.IMGS_DIR /home/yihuihe/outs NUM_DEVICES 1 TRAIN.BATCH_SIZE 200 TEST.BATCH_SIZE 200 "$@"
  fi
elif [[ $action == assign ]] ; then
  if [[ $place == cluster ]] ; then
    NAME=$name \
    TYPE=test \
    OUTPUT_PREFIX=trainval \
    OUTPUT_DIR=${root} \
    MODEL=$param \
    CFG=$CFG \
    ./deploy/clustervis.sh --action $action --reference_dir /home/yihuihe/outs EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True "$@"
  elif [[ $place == local ]] ; then
    /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CFG --output_dir $root --reference_dir $REFERENCE_DIR --data_type test --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $param EXTRACT.IMGS_DIR /home/yihuihe/outs NUM_DEVICES 1 TRAIN.BATCH_SIZE 200 TEST.BATCH_SIZE 200 "$@"
  fi

elif [[ $action == comparecluster ]] ; then
  if [[ $place == cluster ]] ; then
    NAME=$name \
    TYPE=test \
    OUTPUT_PREFIX=trainval \
    OUTPUT_DIR=${root} \
    MODEL=$param \
    CFG=$CFG \
    ./deploy/clustervis.sh --action $action --reference_dir /home/yihuihe/outs EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True "$@"
  elif [[ $place == local ]] ; then
    /home/yihuihe/self_supervision_benchmark/tools/extract_features.py --config_file $CFG --output_dir $root --reference_dir $REFERENCE_DIR --data_type test --action $action EXTRACT.KMEANS_K 10000 EXTRACT.PREPROCESS True TEST.PARAMS_FILE $param EXTRACT.IMGS_DIR /home/yihuihe/outs NUM_DEVICES 1 TRAIN.BATCH_SIZE 200 TEST.BATCH_SIZE 200 "$@"
  fi
fi
