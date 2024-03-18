#data:24/3/13
# conda init bash
# conda activate base
#data:24/1/26
# conda init bash
# conda activate base
declare -a Task
#0 is pde task
#1 is exprs folder 
Task[0]=Poisson
full_path=$0
shape="t"
mu="1"
expr_folder="shell/expr_2_11_p_${shape}_mu_${mu}_sub_9_data_driven/MOE/"


export DDE_BACKEND=pytorch

yaml_name=$(basename "$full_path" .sh)

script_name=$(basename "$full_path" .sh)
yaml_path=${expr_folder}${script_name}.yaml
echo "The name of this script is: $script_name"

# #possion
python Expr1_P.py --config_yaml $yaml_path --save_folder $script_name