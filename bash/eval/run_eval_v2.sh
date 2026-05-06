timestamp=$(date +"%Y%m%d_%H%M%S")
output_dir="local/profile/nsys_e_gate_${timestamp}"

nsys profile \
 --trace=cuda,nvtx,osrt \
 --sample=none \
 -o ${output_dir} \
 /home/june/anaconda3/envs/rosetta/bin/python \
 script/evaluation/unified_evaluator_v2.py --config recipe/eval_recipe/unified_eval.yaml




#python script/evaluation/unified_evaluator_v2.py --config recipe/eval_recipe/unified_eval.yaml