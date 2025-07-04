#!/bin/bash
#SBATCH --time=55:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=2
source /users/PAS2062/delijingyic/project/morph/.pitzermorphenv/bin/activate
cwd=$(pwd)

lang=$1
arch=${2:-tagtransformer}
suff=$3

lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
bs=400 # 256

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
#dropout=${2:-0.3}
dropout=0.3
ckpt_dir=checkpoints/sig22

case "$lang" in

*) trn_path=/users/PAS2062/delijingyic/project/morph/neural-transducer/2022InflectionST/part1/development_languages ;;
#*) trn_path=2021Task0/part1/development_languages ;;
esac
# tst_path=../../../part1/submissions/baseline/wuetal-transformer
tst_path=/users/PAS2062/delijingyic/project/morph/neural-transducer/2022InflectionST/part1/development_languages
#tst_path=2021Task0/part1/ground-truth

python3 src/train.py \
    --dataset sigmorphon17task1 \
    --train $trn_path/$lang"_"$suff.train \
    --dev $trn_path/$lang.dev \
    --test $tst_path/$lang.test \
    --model $ckpt_dir/$arch/$lang"_"$suff \
    --decode greedy --max_decode_len 32 \
    --embed_dim $embed_dim --src_hs $hs --trg_hs $hs --dropout $dropout --nb_heads $nb_heads \
    --label_smooth $label_smooth --total_eval $total_eval \
    --src_layer $layers --trg_layer $layers --max_norm 1 --lr $lr --shuffle \
    --arch $arch --gpuid 0 --estop 1e-8 --bs $bs --max_steps $max_steps \
    --scheduler $scheduler --warmup_steps $warmup --cleanup_anyway --beta2 $beta2 --bestacc
