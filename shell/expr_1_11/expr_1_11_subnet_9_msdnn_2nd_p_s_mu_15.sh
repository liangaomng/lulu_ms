#data:24/1/26
# conda init bash
# conda activate base
#data:24/1/26
# conda init bash
# conda activate base
declare -a Task
yaml=shell/expr_1_11/expr_1_11_subnet_9_msdnn_2nd_p_s_mu_15.yaml
#0 is pde task
#1 is exprs folder 
Task[0]=Poisson
full_path=$0
# 使用basename获取不包括路径的文件名

script_name=$(basename "$full_path" .sh)
echo "The name of this script is: $script_name"

#根据这个 来写入到不同的文件夹script_name
#possion
python Expr1_P.py --config_yaml $yaml --save_folder $script_name