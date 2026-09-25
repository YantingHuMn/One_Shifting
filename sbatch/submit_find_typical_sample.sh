#!/bin/bash
#SBATCH --job-name=typical
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --output=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/submit_average.out
#SBATCH --error=/dcs07/hongkai/data/yhu1/One_Shifting_Results/logs/submit_average.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --dependency=afterok:33324229


# module load conda_R

# rm -f /dcs07/hongkai/data/yhu1/One_Shifting_Results/col_reverse_representative_sample_ranking.csv
# Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/typical_sample_find.R 

cd /dcs10/hongkai/data/yhu1/One_Shifting
bash ../One_Shifting/inst/scripts/run_scVI_DCA_Saver.sh \
    "/dcs07/hongkai/data/yhu1/One_Shifting_Results/HCA/PF_specimen_INPUT_RNA" \
    "PF_specimen_RNA" \
    "PF_specimen_ATAC" \
    "zinb"
