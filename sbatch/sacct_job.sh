sacct -j 35510044_1 --format=JobID,JobName%100,State,Elapsed,Start,End
squeue -j 35820817_0 -o "%.20i %.50j"
scontrol show job 35510038_1 | grep TimeLimit


export sample_name="ENCSR196JUO"
export folder_name="ENCODE"

cd /dcs10/hongkai/data/yhu1/One_Shifting
bash /dcs10/hongkai/data/yhu1/One_Shifting/submit_all_steps_8.sh \
    inst/config/config_run_multi_RNA_HCA_10x_ENCODE.sh \
    scVI_mse \
    ""

folder_name="10x"
Rscript /dcs10/hongkai/data/yhu1/One_Shifting/R/check_raw_pair_alignment.R \
    /dcs07/hongkai/data/yhu1/One_Shifting_Results/${folder_name} \
    /dcs07/hongkai/data/yhu1/One_Shifting_Results/${folder_name}_raw_pair_alignment.csv