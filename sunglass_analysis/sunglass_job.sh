#!/bin/bash

#PBS -S /bin/sh
#PBS -N rlc
#PBS -j oe
#PBS -l nodes=1:ppn=1,walltime=3:00:00
# PBS -o /data101/makinen/borg_sunglass/joblogs/outlog.log


module purge
module load cuda/11.8
module load gcc/11.3.0
module load cmake
module load inteloneapi/2023.0
module load intelpython/3-2022.2.1

 
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS
 
 
source /home/makinen/venvs/fastjax/bin/activate

cd /home/makinen/repositories/alfi_lensing/sunglass_analysis/ #/data101/makinen/borg_sunglass/


python3 run_sunglass.py fiducial 1

exit 0
