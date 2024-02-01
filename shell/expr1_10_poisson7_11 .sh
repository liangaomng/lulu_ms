
#data:24/1/30
#固定神经网络中间900 2exp law
# conda init bash
# conda activate base
declare -a Task

#0 is pde task
#1 is exprs folder 
Task[0]=Poisson
Task[1]="Expr2d_1_10"

export DDE_BACKEND=pytorch
echo $TASK

export sv_sh_folder=EXPRS/${Task[0]}/${Task[1]}

echo "cp to path:$sv_sh_folder"

cp $0 "$sv_sh_folder"

# 检查 cp 命令是否成功
if [ $? -eq 0 ]; then
    echo "cp sussess $sv_sh_folder"
else
    echo "cp fail"
fi

for name_order in {7..11}; do

   export expr_floder=EXPRS/${Task[0]}/${Task[1]}/${Task[0]}_Seed_Scale_${Task[1]}_$name_order


   for seed in {1..3};do

      echo $expr_floder/Expr_$seed.xlsx 
      
      python Expr1_run.py --expr_set_path $expr_floder/Expr_$seed.xlsx --pde_name "${Task[0]}" --mu 15.0 --epoch 10000 --fig_record_intereve 500 
      echo "expr: done seed:$seed  order:$name_order"

   done

done