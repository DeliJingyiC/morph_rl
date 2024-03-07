#!/bin/bash
#SBATCH --job-name=multi_x
#SBATCH --account=PAS1957
#SBATCH --output=output/%x.out
#SBATCH --error=output/%x.err
#SBATCH --mem=48000
#SBATCH --time=18:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu_cmode=exclusive

echo $SLURM_JOB_ID
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_NTASKS
#set -ex #prints a bunch of crap

# if test -z $SLURM_JOB_ID; then
#     export SLURM_JOB_ID=$(date +%s)
#     echo "then $SLURM_JOB_ID"
# fi
mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

#module load python/3.9-2022.05
#module load cuda
source activate torch5
#cwd=$(pwd)

lang=$1
devset=$2
arch=tagtransformer

lr=0.001
scheduler=warmupinvsqr
#max_steps=20000
max_steps=0 #set so that epochs flag actually does something
epochs=3

warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=128 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=0.3

data_dir=$SLURM_SUBMIT_DIR/data/reinf_inst
ckpt_dir=$SLURM_SUBMIT_DIR/checkpoints/$1

python -u $SLURM_SUBMIT_DIR/example/transformer/src/infer.py \
    --dataset sigmorphon17task1 \
    --train $data_dir/$lang-train \
    --dev $data_dir/$devset \
    --test $data_dir/$lang-test \
    --model $ckpt_dir/$arch/sigmorphon17-task1-dropout$dropout/$lang-$decode \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc --epochs $epochs \
	--load_previous $SLURM_SUBMIT_DIR/checkpoints/ud_UD_Turkish-Kenet_synthetic_char/tagtransformer/sigmorphon17-task1-dropout0.3/ud_UD_Turkish-Kenet_synthetic_char-
