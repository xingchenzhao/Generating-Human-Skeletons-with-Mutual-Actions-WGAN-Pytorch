#!/bin/bash
#SBATCH --job-name=run_gan
#SBATCH --nodes=1 #number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu # mpi, gpu and smp are available in H2P
#SBATCH --partition=gtx1080 # available: smp, high-mem, opa, gtx1080, titanx, k40
#SBATCH --gres=gpu:1
#SBATCH --mem=12000
#SBATCH --mail-user=zig9@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-05:00:00 
module purge #make sure the modules environment is sane
module load python/3.7.0 cuda/10.1 venv/wrap
workon pytorch

python vae0_train.py > vae_result.txt
python gan0_train.py > gan_result.txt
