#!/bin/bash
#SBATCH --partition=insa-gpu
#SBATCH --job-name=MERS-REST-vit21
#SBATCH --output=/calcul-crn21/amouath/output_21.txt
#SBATCH --error=/calcul-crn21/amouath/error_21.txt
#SBATCH -w crn21
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type = ALL
#SBATCH --mail-user = aouayeb.mouath@insa-rennes.fr

srun singularity run --nv --bind /calcul-crn21/amouath/cluster_gpu/MERS1/content/ --bind /calcul-crn21/amouath/cluster_gpu/MERS1/FER-transformer-master-src/ /calcul-crn21/amouath/cluster_gpu/MERS1/singularity/image.sif /bin/bash -c "python3 run.py"
