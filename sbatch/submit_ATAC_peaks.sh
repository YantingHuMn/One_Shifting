#!/bin/bash
#SBATCH --job-name=ATAC_peaks
#SBATCH --partition=gpu     
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=400G
#SBATCH --output=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_all_steps_vae_1_%a.out
#SBATCH --error=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_all_steps_vae_1_%a.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3

methods=("VAE" "DCA_mse" "scVI_mse" "Transformer_denoise")
# methods=("VAE" "scVI_mse")
method=${methods[$SLURM_ARRAY_TASK_ID]}

cd /dcs10/hongkai/data/yhu1/One_Shifting

bash ../One_Shifting/inst/scripts/run_multiomics_general_select_norm_peak.sh \
    ../One_Shifting/inst/config/config_run_multiomics_general_select_norm_peak.sh \
    "$method"

bash ../One_Shifting/inst/scripts/run_bubble_method_plot.sh \
    ../One_Shifting/inst/config/config_run_multiomics_general_select_norm_peak.sh \
    ""

bash ../One_Shifting/inst/scripts/run_bar_method_plot.sh \
    ../One_Shifting/inst/config/config_run_multiomics_general_select_norm_peak.sh \
    ""
