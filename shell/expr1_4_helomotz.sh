#实验1_4
#data:24/1/21
# conda init bash
# conda activate base
export DDE_BACKEND=pytorch
export pde_task=Helmholtz

python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_4/Expr_1.xlsx" --pde_name "$pde_task" --wave_number 1



