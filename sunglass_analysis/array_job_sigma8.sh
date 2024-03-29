#PBS -N sunglass_s8
#PBS -q batch
#PBS -l walltime=3:00:00
#PBS -t 0-200
#PBS -j oe
#PBS -o ${HOME}/data/jobout/sunglass/sigma8/${PBS_JOBNAME}.${PBS_JOBID}.o

#PBS -l nodes=1:ppn=1,walltime=2:30:00,mem=32GB

echo cd-ing...
cd /home/makinen/repositories/alfi_lensing/sunglass_analysis/

echo activating environment...

module purge
module load cuda/11.8
module load gcc/11.3.0
module load cmake
module load inteloneapi/2023.0
module load intelpython/3-2022.2.1


XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

source /home/makinen/venvs/fastjax/bin/activate

echo running script...
echo "arrayind is ${PBS_ARRAYID}"
python run_sunglass.py sigma8 ${PBS_ARRAYID} 5

echo done...


exit 0
