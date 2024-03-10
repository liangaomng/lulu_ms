
#data:24/1/30
#固定神经网络中间900 2exp law
# conda init bash
# conda activate base
declare -a Task

#0 is pde task
#1 is exprs folder 
Task[0]=Poisson
Task[1]="Expr2d_moe_false_3_2"

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

for name_order in 1; do

   export expr_floder=EXPRS/${Task[0]}/${Task[1]}


   for seed in 1;do

      echo $expr_floder/Expr_$seed.xlsx 
      
      python Expr1_run.py --expr_set_path $expr_floder/Expr_$seed.xlsx --pde_name "${Task[0]}" --mu 30.0 --epoch 10010 --fig_record_intereve 50 
      echo "expr: done seed:$seed  order:$name_order"

   done

done