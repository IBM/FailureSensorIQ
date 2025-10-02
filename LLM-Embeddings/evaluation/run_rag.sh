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
jbsub "python3 rag.py google-bert/bert-base-uncased bert_base" 1 bert_base.out bert_base.err bert_base
jbsub "python3 rag.py ../models/google-bert_bert-base-uncased_fmsr_ft_bs32_poolmean_pdesc0.0_positiveratio_0.5seed_42/checkpoint-4900 bert_ft" 1 bert_ft.out bert_ft.err bert_ft
jbsub "python3 rag.py sentence-transformers/all-mpnet-base-v2 mpnet_base" 1 mpnet_base.out mpnet_base.err mpnet_base
jbsub "python3 rag.py ../models/sentence-transformers_all-mpnet-base-v2_fmsr_ft_bs32_poolmean_pdesc0.4_positiveratio_0.5seed_42/checkpoint-9600 mpnet_ft" 1 mpnet_ft.out mpnet_ft.err mpnet_ft
jbsub "python3 rag.py BAAI/bge-large-en-v1.5 baai_base" 1 baai_base.out baai_base.err baai_base
jbsub "python3 rag.py ../models/BAAI_bge-large-en-v1.5_fmsr_ft_bs32_poolmean_pdesc0.4_positiveratio_0.5seed_42/checkpoint-8600 baai_ft" 1 baai_ft.out baai_ft.err baai_ft