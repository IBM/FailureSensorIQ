models="ibm-granite/granite-3.0-8b-instruct ibm-granite/granite-3.2-8b-instruct ibm-granite/granite-3.3-8b-instruct meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b microsoft/phi-4 qwen/qwen2.5-7b-instruct deepseek-ai/DeepSeek-R1-Distill-Qwen-7B deepseek-ai/DeepSeek-R1-Distill-Llama-8B meta-llama/llama-3.3-70b-instruct mistralai/Mixtral-8x7B-v0.1 mistralai/Mistral-Large-Instruct-2407 mistralai/Mixtral-8x22B-v0.1"
for model in $models; do
    model="$(echo $model | cut -d'/' -f2)"
    echo "$model"
    python uncertainty_quantification_via_cp.py   --model=${model}   --raw_data_dir=data   --logits_data_dir=outputs_chat_v2   --data_names=fmsr   --cal_ratio=0.5   --icl_methods=icl0   --prompt_methods=base   --alpha=0.1 
done

