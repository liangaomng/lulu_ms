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

python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_3/Expr_3.xlsx" --pde_name "$pde_task" --wave_number 1.0
python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_3/Expr_4.xlsx" --pde_name "$pde_task" --wave_number 1.0

#for test k=100 高波数
python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_3/Expr_5.xlsx" --pde_name "$pde_task" --wave_number 100.0
