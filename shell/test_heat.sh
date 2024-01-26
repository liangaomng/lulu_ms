export DDE_BACKEND=pytorch
export pde_task=Heat
export sub_folder=Expr2d_1_4

for name_order in {1..10}; do

   export expr_floder="EXPRS/$pde_task/$sub_folder"

   for seed in {1..3};do

      echo $expr_floder/Expr_$seed.xlsx 
      
      python Expr1_run.py --expr_set_path $expr_floder/Expr_1.xlsx --pde_name "$pde_task" --mu 15.0 --epoch 100 --fig_record_intereve 500 
      echo "expr: done seed:$seed  order:$name_order"

   done

done