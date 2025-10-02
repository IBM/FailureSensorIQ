

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

jupyter nbconvert --to python training/multi_task_varying_positive.ipynb --output-dir training
use_ccc=$1
declare -a seed=(42 43 44 45 46)
declare -a pratio=(0.000 0.125 0.250 0.375 0.500 0.625 0.750 0.875 1.000)
mkdir -p logs
for i in "${seed[@]}"
do
   for j in "${pratio[@]}"
   do
      # jbsub -q nonstandard -cores 1x8+1 -mem 128G -err logs/bert_err.log -out logs/bert_out.log -name bert_emb -r "a100_80gb" torchrun --standalone --nnodes=1 --nproc-per-node=4 multi_task_inbatch_neg.py google-bert/bert-base-uncased 32 mean "$j" 0.5 "$i" False
      cd training
      if [ "${use_ccc}" = "true" ]; then
         echo "running on ccc"
         cmd="torchrun --standalone --nnodes=1 --nproc-per-node=1 multi_task_varying_positive.py google-bert/bert-base-uncased 32 mean 0.0 "$j" "$i" False"
         out_f="../logs/pdesc${j}_seed${i}.out" 
         err_f="../logs/pdesc${j}_seed${i}.err"
         jbname="bert_${i}"
         jbsub "${cmd}" 1 ${out_f} ${err_f} ${jbname}
      else
         echo "running locally"
         torchrun --standalone --nnodes=1 --nproc-per-node=1 multi_task_varying_positive.py google-bert/bert-base-uncased 32 mean 0.0 "$j" "$i" False
      fi
      cd ..
   done
done

