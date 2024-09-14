if [ $# == 0 ] 
then
    SEED=44
    LR=2e-5
    BatchSize=4
else
    SEED=$1
    LR=$2
fi

work_path=exp/mlee_large/$SEED
mkdir -p $work_path

CUDA_VISIBLE_DEVICES=0 python -u engine.py \
    --model_type=paie \
    --dataset_type=MLEE \
    --model_name_or_path=bart-large \
    --role_path=./data/MLEE/MLEE_role_name_mapping.json \
    --prompt_path=./data/prompts/prompts_MLEE_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --batch_size=4 \
    --max_steps=10000 \
    --max_enc_seq_length 500 \
    --max_dec_seq_length 100 \
    --bipartite 

