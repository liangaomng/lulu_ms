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

# 获取除了最后的 ".sh" 后缀之外的部分
# 使用字符串的分割和切片操作提取 ".sh" 前面的部分
expr_folder="${full_path%.sh}"
echo $expr_folder
export DDE_BACKEND=pytorch

yaml_name=$(basename "$full_path" .sh)

script_name=$(basename "$full_path" .sh)
yaml_path=${expr_folder}/${script_name}.yaml
echo "The name of this script is: $script_name"
echo "yaml:$yaml_path"
# # #possion
python Expr1_P.py --config_yaml $yaml_path --save_folder $script_name