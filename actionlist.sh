#ACTION=svmtest LIST=exp/extract_conv5_ycmm_scratch_true.txt bash extract_all.sh
#ACTION=svmtest LIST=exp/conv4_init_yfcc.txt bash extract_all.sh
#ACTION=svmtest LIST=exp/conv4_scratch_yfcc.txt bash extract_all.sh
#ACTION=svmtest LIST=exp/conv5_init_in1k.txt bash extract_all.sh
#ACTION=svmtest LIST=exp/conv5_init_yfcc.txt bash extract_all.sh

# ACTION=svmtest LIST=exp/conv5_scratch_yfcc.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv5_scratch_in1k.txt bash extract_all.sh

#ACTION=extract LIST=exp/conv5_scratch_in1k.txt bash extract_all.sh

#ACTION=extract LIST=exp/conv4_in1k_scratch_bnfreeze_pca.txt bash extract_all.sh
#ACTION=extract LIST=exp/conv4_in1k_init_bnfreeze_pca.txt bash extract_all.sh
#ACTION=extract LIST=exp/conv4_yfcc_init_bnfreeze_pca.txt bash extract_all.sh
#ACTION=extract LIST=exp/conv4_yfcc_scratch_bnfreeze_pca.txt bash extract_all.sh
# ACTION=extract LIST=exp/conv5_in1k_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=extract LIST=exp/conv5_in1k_scratch_bnfreeze_pca.txt bash extract_all.sh
# ACTION=extract LIST=exp/conv5_yfcc_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=extract LIST=exp/conv5_yfcc_scratch_bnfreeze_pca.txt bash extract_all.sh
#
# sleep 1h
# ACTION=svmtest LIST=exp/conv4_in1k_scratch_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv4_in1k_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv4_yfcc_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv4_yfcc_scratch_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv5_in1k_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv5_in1k_scratch_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv5_yfcc_init_bnfreeze_pca.txt bash extract_all.sh
# ACTION=svmtest LIST=exp/conv5_yfcc_scratch_bnfreeze_pca.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/voc2007_recluster
# ROOT=$root ACTION=extract LIST=exp/conv4_yfcc_scratch_pca_centroid.txt bash extract_all.sh
# ROOT=$root ACTION=svm LIST=exp/conv4_yfcc_scratch_pca_centroid.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/voc2007_recluster12
# ROOT=$root ACTION=extract LIST=exp/conv4_yfcc_scratch_pca_recluster2_1crop.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv4_yfcc_scratch_pca_recluster2_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_in1k_scratch_bnfreeze_pca_recluster.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_in1k_scratch_pca_recluster2_1crop.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_in1k_scratch_pca_recluster2_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_yfcc_scratch_bnfreeze_pca_recluster.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_yfcc_scratch_pca_recluster1_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_yfcc_scratch_pca_recluster2.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_yfcc_scratch_pca_recluster2_5crop.txt bash extract_all.sh
#
# sleep 1h

# ROOT=$root ACTION=svmtest LIST=exp/conv4_yfcc_scratch_pca_recluster2_1crop.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv4_yfcc_scratch_pca_recluster2_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_in1k_scratch_bnfreeze_pca_recluster.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_in1k_scratch_pca_recluster2_1crop.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_in1k_scratch_pca_recluster2_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_yfcc_scratch_bnfreeze_pca_recluster.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_yfcc_scratch_pca_recluster1_5crop.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_yfcc_scratch_pca_recluster2.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_yfcc_scratch_pca_recluster2_5crop.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/voc2007_initrecluster
# ROOT=$root ACTION=extract LIST=exp/conv4_init_scratch_pca_centroid.txt bash extract_all.sh
# sleep 1h
# ROOT=$root ACTION=svm LIST=exp/conv4_init_scratch_pca_centroid.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/conv4_init_scratch_pca_centroid_copy
# ROOT=$root ACTION=extract LIST=exp/conv4_init_scratch_pca_centroid-copy.txt bash extract_all.sh
# sleep 1h
# ROOT=$root ACTION=svm LIST=exp/conv4_init_scratch_pca_centroid-copy.txt bash extract_all.sh

# TODO

# # root=/home/yihuihe/caffe2/resnet/gen/conv4_init_scratch_pca_centroid_success
# # ROOT=$root ACTION=extract LIST=exp/conv4_init_scratch_pca_centroid-copy-success.txt bash extract_all.sh
# # sleep 1h
# # ROOT=$root ACTION=svm LIST=exp/conv4_init_scratch_pca_centroid-copy-success.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv4_init_scratch_pca_centroid-copy-success.txt bash extract_all.sh


# root=/home/yihuihe/caffe2/resnet/gen/pca_long
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_scratch_pca_long.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/centroid
# ROOT=$root ACTION=extract LIST=exp/conv4_yfcc_init_pca_centroid.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/5crop_nopred
# ROOT=$root ACTION=extract LIST=exp/conv4_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv4_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=extract LIST=exp/conv5_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
#
# sleep 1h
#
# root=/home/yihuihe/caffe2/resnet/gen/pca_long
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_scratch_pca_long.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/centroid
# ROOT=$root ACTION=svm LIST=exp/conv4_yfcc_init_pca_centroid.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/5crop_nopred
# ROOT=$root ACTION=svm LIST=exp/conv4_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svm LIST=exp/conv4_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svm LIST=exp/conv5_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svm LIST=exp/conv5_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/pca_long
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv4_in1k_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv4_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_init_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv5_yfcc_scratch_pca_long.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longiter.txt LIST=exp/conv5_in1k_scratch_pca_long.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/centroid
# ROOT=$root ACTION=svmtest LIST=exp/conv4_yfcc_init_pca_centroid.txt bash extract_all.sh
#
# root=/home/yihuihe/caffe2/resnet/gen/5crop_nopred
# ROOT=$root ACTION=svmtest LIST=exp/conv4_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv4_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_in1k_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest LIST=exp/conv5_yfcc_scratch_pca_recluster2_5crop_nopred.txt bash extract_all.sh





# root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_all.sh
# sleep 1h
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_all.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_all.sh
# #
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=extract ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_rest.sh
# # sleep 1h

# root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svm ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_rest.sh

root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_all.sh
root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_all.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_all.sh
ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_all.sh

# root=/home/yihuihe/caffe2/resnet/gen/pca_longer_scratchconv5
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_scratchconv5_pca_longer.txt bash extract_rest.sh
# root=/home/yihuihe/caffe2/resnet/gen/pca_longer
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_init_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_init_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_in1k_scratch_pca_longer.txt bash extract_rest.sh
# ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_in1k_scratch_pca_longer.txt bash extract_rest.sh
# # ROOT=$root ACTION=svmtest ITERLIST=exp/longeriter.txt LIST=exp/conv4_yfcc_scratch_pca_longer.txt bash extract_rest.sh
