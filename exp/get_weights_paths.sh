set -e
set -x

list=${LIST-det.txt}
iterlist=${ITERLIST-longeriter.txt}

for line in $(cat $list) ; do
  # for iter in c2_model_iter5005 c2_model_iter10010 bestModel latest ; do
  for iter in $(cat $iterlist) ; do # c2_model_iter50050 c2_model_iter45045
    p=$line/imagenet1k/resnet_midlevel_delayed_fc/depth_50/${iter}.pkl
    if [[ $p != *"bistro/gpu"* ]]; then
      if [[ -e /home/$p ]]; then
        p=/home/$p
        jobpath=/home/$line
      elif [[ -e /home/$p ]]; then
        p=/home/$p
        jobpath=/home/$line
      else
        p=nonexist
      fi
    else
      jobpath=$line
    fi
    if [[ $p != nonexist ]]; then
      # echo $p
      echo $p >> weights_paths.txt
    fi
  done
  echo $jobpath/stdout.log >> log_paths.txt
done
