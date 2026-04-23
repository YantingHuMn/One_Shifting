

module load conda_R

save_dir="../One_Shifting_Results"
mkdir -p $save_dir

h5ad_path="/dcs07/hongkai/data/mjiang/scMultiomics/database/HCA_hyt/processed/PMID_39640563/multi_PMID_39640563_G120_D_TL_DHS_blacklist_rm_noY.h5ad"

Rscript ../One_Shifting/R/load_rna_atac_from_h5ad.R \
    $h5ad_path \
    $save_dir