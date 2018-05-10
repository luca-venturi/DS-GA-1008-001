#!/bin/bash
#SBATCH --job-name=planted_clique
#SBATCH --output=planted_clique.out
#SBATCH --error=planted_clique.err

#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --mem=256000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lv800@nyu.edu

module purge
module load python-3
srun python3 main_experiment.py
