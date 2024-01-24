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
export pde_task=Helmholtz
#python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d/Expr_1.xlsx" --pde_name "$pde_task" --wave_number 1.0
#python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d/Expr_2.xlsx" --pde_name "$pde_task" --wave_number 1.0
python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d/Expr_3.xlsx" --pde_name "$pde_task" --wave_number 1.0
python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d/Expr_4.xlsx" --pde_name "$pde_task" --wave_number 1.0
# python Expr2_run.py --expr_set_path "Expr2d/Expr_5.xlsx"
# python Expr2_run.py --expr_set_path "Expr2d/Expr_6.xlsx"
# python Expr2_run.py --expr_set_path "Expr2d/Expr_7.xlsx"

