set -x
set -e
original=$1
dest=$2
$HOME/dev/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/extra_scripts/pickle_jigsaw_model.py --model_file ${original} --output_file ${dest} --center_patch 4
$HOME/dev/gen/experimental/deeplearning/yihuihe/self_supervision_benchmark/extra_scripts/pickle_caffe2_detection.py --c2_model ${dest} --output_file ${dest} --absorb_std
