tag=${TAG-aicompute}
cfg=${CFG-configs/benchmark_tasks/image_classification/in1k/resnet_extract_features.yaml}
launcher=${LAUNCHER-./deploy/launch.sh}
set -x
set -e
#j=5.0
#k=0.01
#i=8
#TAG=$tag CFG=$cfg NAME=sigma${j}_top${i}_LR${k}_FC16 ./deploy/cluster/launch.sh MIDLEVEL.GAUSSIAN_TOPK $i MIDLEVEL.GAUSSIAN_SIGMA $j SOLVER.BASE_LR $k
for j in 5.0 ; do #5.0 10.0
    for k in 0.1 0.01 ; do #0.1 0.01 0.001
        for i in 8 ; do #2 8 100 500 1000
          TAG=$tag CFG=$cfg NAME=sigma${j}_top${i}_LR${k}_crop5 ${launcher} MIDLEVEL.GAUSSIAN_TOPK $i MIDLEVEL.GAUSSIAN_SIGMA $j SOLVER.BASE_LR $k "$@"
    done
  done
done
    # CFG=configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale.yaml NAME=sigma${j}_LR${k}_FC16 ./deploy/cluster/launch.sh MIDLEVEL.GAUSSIAN_SIGMA $j SOLVER.BASE_LR $k ;
          #CFG=configs/unsupervised/midlevel/pretext/resnet/IN1k_8gpu_delayed_BN_bias_decay_shift_scale.yaml NAME=sigma${j}_top${i}_LR${k}_FC16 ./deploy/cluster/launch.sh MIDLEVEL.GAUSSIAN_TOPK $i MIDLEVEL.GAUSSIAN_SIGMA $j SOLVER.BASE_LR $k ;
