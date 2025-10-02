# SEED=42
# PDESC=0.0
# POSITIVE_RATIO=0.5
# multi_task_training=True
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/mpnet_err.log -out logs/mpnet_out.log -name mpnet_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py sentence-transformers/all-mpnet-base-v2 32 mean $PDESC $POSITIVE_RATIO $SEED $multi_task_training
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/bge_err.log -out logs/bge_out.log -name bge_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py BAAI/bge-large-en-v1.5 32 mean $PDESC $POSITIVE_RATIO $SEED $multi_task_training
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/bert_err.log -out logs/bert_out.log -name bert_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py google-bert/bert-base-uncased 32 mean $PDESC $POSITIVE_RATIO $SEED $multi_task_training
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/qwen_err.log -out logs/qwen_out.log -name qwen_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py Alibaba-NLP/gte-Qwen2-7B-instruct 32 last $PDESC $POSITIVE_RATIO $SEED $multi_task_training
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/e5_err.log -out logs/e5_out.log -name e5_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py intfloat/e5-mistral-7b-instruct 32 mean $PDESC $POSITIVE_RATIO $SEED $multi_task_training
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/granite_err.log -out logs/granite_out.log -name granite_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_embedding.py ibm-granite/granite-3.1-8b-instruct 32 last
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/llm2vec_err.log -out logs/llm2vec_out.log -name llm2vec_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 McGill-NLP/LLM2Vec-Meta-llama-31-8B-Instruct-mntp-supervised 4
# jbsub -q nonstandard -cores 1x8+4 -mem 128G -err logs/nvembed_err.log -out logs/nvembed_out.log -name nvembed_emb torchrun --standalone --nnodes=1 --nproc-per-node=4 nvidia/NV-Embed-v2 4


#!/bin/bash

jbsub() {
	# jbsub ./run.sh 1 out.txt err.txt jobname
	if [ -z "$3" ]; then
		outfile="out.txt"
	else
		outfile="$3"
	fi
	if [ -z "$4" ]; then
		errfile="err.txt"
	else
		errfile="$4"
	fi
	if [ -z "$2" ]; then
		ngpu="1"
	else
		ngpu="$2"
	fi
	if [ -z "$5" ]; then
		jbname="job"
	else
		jbname="$5"
	fi
	bsub -q normal -gpu num=$ngpu:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB -oo $outfile -eo $errfile -J $jbname $1
}
export -f jbsub

jupyter nbconvert --to python training/multi_task_inbatch_neg.ipynb --output-dir training
use_ccc=$1
declare -a seed=(42 43 44 45 46)
declare -a pdesc=(0.0 0.2 0.4 0.6 0.8 1.0)
mkdir -p logs
for i in "${seed[@]}"
do
   for j in "${pdesc[@]}"
   do
      cd training
      if [ "${use_ccc}" = "true" ]; then
         echo "running on ccc"
         cmd="torchrun --standalone --nnodes=1 --nproc-per-node=1 multi_task_inbatch_neg.py google-bert/bert-base-uncased 32 mean $j 0.5 $i False"
         out_f="../logs/pdesc${j}_seed${i}.out" 
         err_f="../logs/pdesc${j}_seed${i}.err"
         jbname="bert_${i}"
         jbsub "${cmd}" 1 ${out_f} ${err_f} ${jbname}
      else
         echo "running locally"
         torchrun --standalone --nnodes=1 --nproc-per-node=1 multi_task_inbatch_neg.py google-bert/bert-base-uncased 32 mean $j 0.5 $i False
      fi
      cd ..
   done
done

