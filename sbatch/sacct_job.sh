sacct -j 35048082_3 --format=JobID,JobName%100,State,Elapsed,Start,End
squeue -j 35048081_0 -o "%.20i %.50j"
scontrol show job 34859152_6 | grep TimeLimit