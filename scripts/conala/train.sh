#!/bin/bash

source activate py2torch3cuda9

seed=${1:-0}
vocab="vocab.freq0.bin"
train_file="train_final.bin"
dev_file="test.bin"
dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
ptrnet_hidden_dim=32
lr=0.001
lr_decay=0.5
beam_size=20
lstm='lstm'  # lstm
model_name=final_subsuq.model.sup.conala.${lstm}.hidden${hidden_size}.embed${embed_size}.action${action_embed_size}.field${field_embed_size}.type${type_embed_size}.dropout${dropout}.lr${lr}.lr_decay${lr_decay}.beam_size${beam_size}.${vocab}.${train_file}.glorot.par_state_w_field_embed.seed${seed}

python exp.py \
    --seed ${seed} \
    --cuda \
    --mode train \
    --batch_size 1 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --train_file data/conala/${train_file} \
    --dev_file data/conala/${dev_file} \
    --vocab data/conala/${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 2 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --valid_every_epoch 1 \
    --lr_decay_after_epoch 8 \
    --beam_size ${beam_size} \
    --log_every 50 \
    --validate_with_bleu 1 \
    --save_to saved_models/conala/${model_name} 2>logs/conala/${model_name}.log
    #--save_to saved_models/conala/${model_name} 

#     --no_parent_state \

. scripts/conala/test.sh saved_models/conala/${model_name}.bin 2>>logs/conala/${model_name}.log
