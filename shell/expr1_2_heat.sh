
# 检查并安装 pandas
#pip install pandas
#pip install openpyxl==3.0.10
#pip install xlrd==1.2.0
#pip install captum
#pip install deepxde
#cd if you need

#实验1
#data:24/1/13
# conda init bash
# conda activate base
export DDE_BACKEND=pytorch
export pde_task=Heat
python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_4/Expr_1.xlsx" --pde_name "$pde_task" --freq 1.0
