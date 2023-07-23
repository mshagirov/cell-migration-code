#!/bin/bash
#PBS -l walltime=04:00:00
#PBS -l select=1:mem=32G:ncpus=16:ngpus=1:host=mala-pbs
#PBS -N nb_test
#PBS -j oe
#PBS -M shagirov@nus.edu.sg
#PBS -m abe
#PBS -P hpc_test

hostname
nvidia-smi
module purge

export PATH=~/.local/bin:$PATH

cd ~/MBIHPC/jupyterjob

XDG_RUNTIME_DIR=""



pbs_id=$(echo $PBS_JOBID| awk -F '.' '{print $1}')

outputfile=${pbs_id}.`hostname`.out

echo -e "\n"   >> $outputfile
echo -e "\n HOSTNAME: `hostname` \n"
echo -e "  \n----- ----- -----\nNVIDIA GPU\n(if CUDA and drivers installed, you should see nvidia-smi output below)" >> $outputfile

nvidia-smi     >> $outputfile
echo -e "\n"   >> $outputfile

source activate fastai
python ./ft_resnet34.py > ${pbs_id}.stdout 2> ${pbs_id}.stdmsg
