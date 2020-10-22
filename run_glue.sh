export GLUE_DIR=./glue_data
export TASK_NAME=SST-2

CUDA_VISIBLE_DEVICES=0 \
python run_glue.py \
  --model_name_or_path roberta-base \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_aug \
  --aug_type 'span_cutoff' \
  --aug_cutoff_ratio 0.1 \
  --aug_ce_loss 1.0 \
  --aug_js_loss 1.0 \
  --learning_rate 5e-6 \
  --num_train_epochs 10.0 \
  --logging_steps 500 \
  --save_steps 500 \
  --per_gpu_train_batch_size 8 \
  --output_dir results/$TASK_NAME-roberta_base-cutoff \
  --overwrite_output_dir
