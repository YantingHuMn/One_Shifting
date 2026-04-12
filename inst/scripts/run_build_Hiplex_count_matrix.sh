
module load conda_R

Rscript ../R/Step1_build_count_matrix.R \
  /dcs05/hongkai/data/next_cutntag/bulk/homotone_heterotone_merged/data_align/frag_decon/valley-all-qc/V1_mixed \
  /dcs05/hongkai/data/next_cutntag/bulk/homotone_heterotone_merged/data_align/frag_decon/valley-all-qc/V2_mixed \
  "$READ_DIR" \
  "$norm_factor_string" \
  "$norm_factor_string"