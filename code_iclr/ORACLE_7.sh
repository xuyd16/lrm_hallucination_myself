#!/bin/bash -l
#SBATCH --job-name=bfm_912M
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=accelerated


nvidia-smi

echo "=== BFM EEG Pre-training Job Started ==="
echo "Job started on $(hostname) at $(date)"
echo "Current working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"


# 获取作业节点列表中的第一个节点作为主节点
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# 设置一个未被占用的端口
export MASTER_PORT=29500

echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

source ~/.bashrc
module load devel/cuda/11.8 


export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# OMP_NUM_THREADS: PyTorch张量运算线程数（16个）
# NUM_WORKERS: 数据加载器多进程数（12个）
# 预留4个CPU用于系统和其他进程
export OMP_NUM_THREADS=2
export NUM_WORKERS=24


export WANDB_MODE="offline"


cd /hkfs/work/workspace/scratch/tum_fmp0582-ictspace/bfm_iclr || exit 1


CONDA_BASE_PATH="/home/hk-project-p0022560/tum_fmp0582/anaconda3"
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate myenv1

mkdir -p logs
mkdir -p checkpoint

echo "=== Environment Setup Complete ==="
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

export DATA_PATH="/hkfs/work/workspace/scratch/tum_fmp0582-ictspace/processed_data2"

echo "=== Starting EEG Model Pre-training ==="


export EMBED_DIM=512              
export NUM_HEADS=16               
export DEPTH=26                  


export BATCH_SIZE=8              
export LEARNING_RATE=3e-5       
export WEIGHT_DECAY=1e-4          
export WARMUP_EPOCHS=10            
export CLIP_GRAD=1.0            



export MASK_RATIO=0.15            
export MASK_STRATEGY="random"    
export MASK_NOISE_RATIO=0.005   


export USE_MOE=True               
export NUM_EXPERTS=16             
export TOP_K_EXPERTS=2            
export MOE_AUX_LOSS_COEFF=0.01    


export FREQ_MASK_RATIO=0.3             


export USE_AMP=True              
export FREQ_EVAL=True            
export DEBUG=False               


export NUM_EPOCHS=100            
export LOG_DIR="./logs"          


EPOCHS=$NUM_EPOCHS


srun python -u train_H100.py 2>&1 | tee "./logs/pretraining_${SLURM_JOB_ID}.log"


if [ $? -eq 0 ]; then
    echo "=== Pre-training completed successfully ==="
else
    echo "=== Pre-training failed with exit code $? ==="
fi


echo "=== Final GPU Status ==="
nvidia-smi


echo "=== Disk Usage ==="
du -sh ./checkpoint
du -sh ./logs

echo "=== Job completed at $(date) ===" 