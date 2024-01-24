#data:24/1/21
# conda init bash
# conda activate base
export DDE_BACKEND=pytorch
export pde_task=Poisson

python Expr1_run.py --expr_set_path "EXPRS/$pde_task/Expr2d_1_5/Expr_1.xlsx" --pde_name "$pde_task" --mu 15.0
