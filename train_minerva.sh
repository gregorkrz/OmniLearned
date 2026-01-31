#omnilearned train   --output_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset    --save-tag M1A_L1   --dataset minerva_1A   --path /global/cfs/cdirs/m3246/gregork/Minerva/20260127_nested_split   --mode regression   --regression-loss l1   --batch 128  --epoch 20   --lr 5e-5   --size small   --wd 0.1   --num-workers 5 --use-pid --wandb --nevts 1000000 

# Train using the pretrained ckpt
#omnilearned train   --output_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset_FT_OmniLearnedSmall    --save-tag M1A_L1_pretrainedSmall   --dataset minerva_1A   --path /global/cfs/cdirs/m3246/gregork/Minerva/20260127_nested_split   --mode regression   --regression-loss l1   --batch 128  --epoch 20   --lr 5e-5   --size small   --wd 0.1   --num-workers 5 --use-pid --wandb --nevts 1000000  --fine-tune --pretrain-tag pretrain_s


#request a single gpu 4 hour session
#salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 1 --account m3246


# EVAL

omnilearned evaluate \
  --input_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset \
  --output_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset/eval_results \
  --save-tag M1A_L1 \
  --dataset minerva_1A \
  --path /global/cfs/cdirs/m3246/gregork/Minerva/20260127_nested_split \
  --mode regression \
  --size small \
  --batch 128 \
  --num-workers 16 \
  --use-pid



  omnilearned evaluate \
  --input_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset_FT_OmniLearnedSmall \
  --output_dir /global/cfs/cdirs/m3246/gregork/checkpoints/20260127_nested_split/L1_initial_training_smalldataset_FT_OmniLearnedSmall/eval_results \
  --save-tag M1A_L1_pretrainedSmall \
  --dataset minerva_1A \
  --path /global/cfs/cdirs/m3246/gregork/Minerva/20260127_nested_split \
  --mode regression \
  --size small \
  --batch 128 \
  --num-workers 16 \
  --use-pid