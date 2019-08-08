set -e
# LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash det_all.sh
# LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash det_all.sh
# LIST=exp/conv5_in1k_init_pca_longer.txt bash det_all.sh
# LIST=exp/conv5_yfcc_init_pca_longer.txt bash det_all.sh
# LIST=exp/conv5_in1k_scratch_pca_longer.txt bash det_all.sh
# LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash det_all.sh
#https://our.internmc.facebook.com/intern/fblearner/runs?query=%7B%22ids%22%3A[130618833%2C130619028%2C130618556%2C130618560%2C130619029%2C130618557%2C130618558%2C130618559%2C130619021%2C130619022%2C130619023%2C130618745%2C130618746%2C130619195%2C130619196%2C130618748%2C130618751%2C130618735%2C130619197%2C130618749%2C130618670%2C130618671%2C130618737%2C130618672%2C130618738%2C130618673%2C130619190%2C130619191%2C130619189%2C130619192%2C130618667%2C130619193%2C130618668%2C130618669%2C130618834%2C130618835%2C130618842%2C130619024%2C130618830%2C130619025%2C130618831%2C130618829%2C130618832]%7D&sort_by=time_created&sort_direction=DESC&grouping=Status

# ITERLIST=exp/longiter.txt LIST=exp/det.txt bash det_all.sh
# result in : https://our.internmc.facebook.com/intern/fblearner/runs?query=%7B%22ids%22%3A[130586004%2C130586005%2C130586006%2C130585994%2C130585996%2C130585997%2C130585984%2C130586000%2C130585985%2C130585999%2C130586001%2C130585914%2C130585986%2C130585928%2C130586002%2C130585932%2C130586003%2C130585988%2C130585992%2C130585980%2C130585981%2C130585982%2C130585983]%7D&esq=%7B%22key%22%3A%22AND%22%2C%22children%22%3A[%7B%22key%22%3A%22CONTAINS_NONE_OF_FBIDS%22%2C%22field%22%3A%22FBLEARNER_tags%22%2C%22value%22%3A[%7B%22title%22%3A%22hide%22%2C%22fbid%22%3A%22230919650311895%22%7D]%7D%2C%7B%22key%22%3A%22EQUALS_ANY_OF_FBLEARNER_RUN_OWNER%22%2C%22field%22%3A%22FBLEARNER_owner_unixname%22%2C%22value%22%3A[%7B%22title%22%3A%22Yihui%20He%22%2C%22fbid%22%3A%22100027927526263%22%7D]%7D%2C%7B%22key%22%3A%22IS_NOT_SET%22%2C%22field%22%3A%22FBLEARNER_parent%22%2C%22value%22%3Anull%7D%2C%7B%22key%22%3A%22EQUALS_ANY_OF_FBLEARNER_RUN_STATUS%22%2C%22field%22%3A%22FBLEARNER_run_status%22%2C%22value%22%3A[%222%22%2C%221%22]%7D]%7D&sort_by=time_created&sort_direction=DESC&grouping=Status

# ITERLIST=exp/shortiter.txt LIST=exp/det.txt bash det_all.sh

#imagenet classification
LAUNCHER=placescls.sh LIST=exp/conv5_yfcc_init_scratchconv5_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh LIST=exp/conv5_in1k_init_scratchconv5_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh LIST=exp/conv5_in1k_init_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh LIST=exp/conv5_yfcc_init_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh LIST=exp/conv5_in1k_scratch_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh LIST=exp/conv5_yfcc_scratch_pca_longer.txt bash det_all.sh
LAUNCHER=placescls.sh ITERLIST=exp/longiter.txt LIST=exp/det.txt bash det_all.sh
