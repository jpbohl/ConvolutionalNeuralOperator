#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --tmp=10G
#SBATCH --mem-per-cpu=512
#SBATCH --job-name=CNO_Wave
#SBATCH --ntasks=16
#SBATCH --gpus=1


module load gcc/8.2.0 python_gpu/3.11.2
module load cuda/11.7.0

python3 TrainCNO.py /cluster/scratch/jabohl/TestCNO/CNO_Wave training_properties.json model_architecture.json wave_0_5 /cluster/scratch/jabohl/CNOData/
