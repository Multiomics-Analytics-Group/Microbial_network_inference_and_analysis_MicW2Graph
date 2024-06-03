#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J cclasso0_1_wwt_assembly_speciesMAN
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=20GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 20GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 120:00
### Select CPU model
#BSUB -R "select[avx512]" 
### -- set the email address -- 
#BSUB -u seayal@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /work3/seayal/MANs_species/Bash_scripts/Biomes_exptypes/Wastewater_assembly/log/Output_%J.out 
#BSUB -e /work3/seayal/MANs_species/Bash_scripts/Biomes_exptypes/Wastewater_assembly/log/Output_%J.err 

### Load R module
module load R/4.3.1-mkl2023update1

### setting the working directory
PROJECT_DIR=/work3/seayal/MANs_species

### Run script
Rscript ${PROJECT_DIR}/Build_microb_net_species_bybiome_and_exptype.R --threshold 0.1 --network cclasso --biome_exptype Wastewater_assembly