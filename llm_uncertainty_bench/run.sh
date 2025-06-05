# model_id="meta-llama/Llama-3.1-8B-Instruct"
model_id=$1
prompt_type=$2
dataset=$3
n_gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
model_name="$(echo $model_id | cut -d'/' -f2)"
echo ${model_name}
echo ${prompt_type}
echo ${n_gpus}
#    torchrun --nproc-per-node ${n_gpus} --master_port 25903 generate_logits_chat.py \
#    torchrun --nproc-per-node ${n_gpus} --master_port 25903 generate_logits.py \

if [ "${prompt_type}" == "chat" ]; then
  echo "chat logits"
  python generate_logits_chat.py --model=${model_id} --data_path=data --file="fmsr_${dataset}.json" --prompt_method=base --output_dir=outputs_chat_v2 --few_shot=0
else
  echo "text generation logits"
  python generate_logits.py --model=${model_id} --data_path=data --file="fmsr_${dataset}.json" --prompt_method=base --output_dir=outputs_chat_v2 --few_shot=0
fi

python uncertainty_quantification_via_cp.py \
  --model=${model_name} \
  --raw_data_dir=data \
  --logits_data_dir=outputs_chat_v2 \
  --data_names=fmsr \
  --cal_ratio=0.5 \
  --icl_methods=icl0 \
  --prompt_methods=base \
  --alpha=0.1 

# fmsr_Acc: 31.15
# Average acc: 31.15
# fmsr_SS: 3.86
# Average SS: 3.86
# fmsr_Coverage Rate: 91.36
# Average Coverage Rate: 91.36
# fmsr_UAcc: 19.75
# Average UAcc: 19.75
