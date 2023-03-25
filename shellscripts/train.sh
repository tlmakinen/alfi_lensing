
#PBS -S /bin/bash
#PBS -N train_imnn
#PBS -j oe
#PBS -o imnnrun.log
#PBS -n
#PBS -l nodes=1:has1gpu:ppn=40,walltime=24:00:00
#PBS -M l.makinen21@imperial.ac.uk

module load tensorflow/2.10
XLA_FLAGS=--xla_gpu_cuda_data_dir=\${CUDA_PATH}
export XLA_FLAGS

source /data80/makinen/venvs/testjax/bin/activate

cd /home/makinen/repositories/alfi_lensing/

# configs, load_model, model_name
python imnn_scripts/train_imnn.py configs/configs_twoparam.json 1 IMNN_w.pkl 123

# make diagnostic plots 
#python diagnostic_plots.py twoparam