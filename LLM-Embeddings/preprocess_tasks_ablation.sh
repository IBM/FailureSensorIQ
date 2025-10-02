jupyter nbconvert --to python preprocessing/preprocess_all_tasks.ipynb --output-dir preprocessing
declare -a seed=(42 43 44 45 46)
declare -a pdesc=(0.0 0.2 0.4 0.6 0.8 1.0)
for i in "${seed[@]}"
do
   for j in "${pdesc[@]}"
   do
     cd preprocessing
     python3 preprocess_all_tasks.py "$j" "$i"
     cd ..
   done
done

