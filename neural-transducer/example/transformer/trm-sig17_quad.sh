#!/bin/bash
#SBATCH --time=10:30:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --gpus-per-node=4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH -p gpuparallel-quad
#SBATCH --gpu_cmode=exclusive

echo $SLURM_JOB_ID
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_NTASKS
set -ex

# if test -z $SLURM_JOB_ID; then
#     export SLURM_JOB_ID=$(date +%s)
#     echo "then $SLURM_JOB_ID"
# fi
mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

source /users/PAS2062/delijingyic/project/morph/.pitzermorphenv/bin/activate
cwd=$(pwd)

lang=$1
arch=tagtransformer

res=high
lr=0.001
scheduler=warmupinvsqr
max_steps=20000
# max_steps=0

warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=256 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${2:-0.3}

data_dir=/users/PAS2062/delijingyic/project/morph/neural-transducer/data
ckpt_dir=/users/PAS2062/delijingyic/project/morph/neural-transducer/checkpoints/transformer

python3 /users/PAS2062/delijingyic/project/morph/neural-transducer/example/transformer/src/train.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/conll2017/all/task1/$lang-train-$res \
    --dev $data_dir/conll2017/all/task1/$lang-dev \
    --test $data_dir/conll2017/answers/task1/$lang-uncovered-test \
    --model $ckpt_dir/$arch/sigmorphon17-task1-dropout$dropout/$lang-$res-$decode \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc --epoch 2
