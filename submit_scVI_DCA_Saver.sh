#!/bin/bash
#SBATCH --job-name=scVI_DCA_Saver
#SBATCH --partition=gpu     
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_scVI_DCA_Saver.out
#SBATCH --error=/dcs10/hongkai/data/yhu1/One_Shifting_Results/logs/submit_scVI_DCA_Saver.err
#SBATCH --mail-user=yhu157@jh.edu
#SBATCH --mail-type=END,FAIL
# #SBATCH --array=0-3

method="MLP"

cd /dcs10/hongkai/data/yhu1/One_Shifting

READ_DIR="../One_Shifting_Results/PBMC_INPUT_RNA"
V1="RNA"
V2="ATAC"
LOSS="zinb"

bash ../One_Shifting/inst/scripts/run_scVI_DCA_Saver.sh \
    $READ_DIR \
    $V1 \
    $V2 \
    $LOSS

