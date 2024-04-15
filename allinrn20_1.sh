#!/bin/bash
# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=10:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova    # gpu node(s)
#SBATCH --mail-user=nsaadati@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load miniconda3
source activate dimat 
./main.sh --nums_models "5" --data_dists "iid" --exps "1" --datasets_nums_classes "cifar100,100" --archs "resnet20" --random_seeds "1" --nums_epochs "2" --num_parallel 1 --merg_itrs 3 --matching_algs "DIMAT"   --run_python true &&
./main.sh --nums_models "20 10 5" --data_dists "iid" --exps "1" --datasets_nums_classes "cifar100,100" --archs "resnet20" --random_seeds "4 3 2 1 0" --nums_epochs "2" --num_parallel 5 --merg_itrs 200 --matching_algs "nonseq_consensus_zip_matching"   --run_python true 
