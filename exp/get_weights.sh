rm weights_paths.txt
rm log_paths.txt
ITERLIST=longeriter.txt LIST=conv5_yfcc_init_scratchconv5_pca_longer.txt bash get_weights_paths.sh
ITERLIST=longeriter.txt LIST=conv5_in1k_init_scratchconv5_pca_longer.txt bash get_weights_paths.sh
ITERLIST=longeriter.txt LIST=conv5_in1k_init_pca_longer.txt bash get_weights_paths.sh
ITERLIST=longeriter.txt LIST=conv5_yfcc_init_pca_longer.txt bash get_weights_paths.sh
ITERLIST=longeriter.txt LIST=conv5_in1k_scratch_pca_longer.txt bash get_weights_paths.sh
ITERLIST=longeriter.txt LIST=conv5_yfcc_scratch_pca_longer.txt bash get_weights_paths.sh

ITERLIST=longiter.txt LIST=conv5_in1k_init_pca_long.txt bash get_weights_paths.sh
ITERLIST=longiter.txt LIST=conv5_yfcc_init_pca_long.txt bash get_weights_paths.sh
ITERLIST=longiter.txt LIST=conv5_in1k_scratch_pca_long.txt bash get_weights_paths.sh
ITERLIST=longiter.txt LIST=conv5_yfcc_scratch_pca_long.txt bash get_weights_paths.sh

ITERLIST=shortiter.txt LIST=conv5_in1k_init_bnfreeze_pca.txt bash get_weights_paths.sh
ITERLIST=shortiter.txt LIST=conv5_yfcc_init_bnfreeze_pca.txt bash get_weights_paths.sh
ITERLIST=shortiter.txt LIST=conv5_in1k_scratch_bnfreeze_pca.txt bash get_weights_paths.sh
ITERLIST=shortiter.txt LIST=conv5_yfcc_scratch_bnfreeze_pca.txt bash get_weights_paths.sh
