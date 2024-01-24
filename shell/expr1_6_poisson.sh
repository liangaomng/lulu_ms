#data:24/1/24
# conda init bash
# conda activate base
export DDE_BACKEND=pytorch
export pde_task=Poisson
export sub_folder=Expr2d_1_6

for name_order in {1..10}; do

   export expr_floder="EXPRS/$pde_task/$sub_folder/Poisson_Seed_Scale_Expr1_6_$name_order"

   for seed in {1..3};do

      echo $expr_floder/Expr_$seed.xlsx 
      
      python Expr1_run.py --expr_set_path $expr_floder/Expr_$seed.xlsx --pde_name "$pde_task" --mu 15.0 --epoch 10000
      echo "expr: done seed:$seed  order:$name_order"

   done

done