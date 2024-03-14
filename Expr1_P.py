import argparse
from src_lam.Agent import Expr_Agent

from src_PDE.Poisson import PDE_PossionData
import yaml 
import numpy as np

if __name__=="__main__":
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Load configuration from YAML")
    parser.add_argument('--config_yaml', type=str, help='Path to the configuration YAML file')
    parser.add_argument('--save_folder', type=str, help='folder name')

    args = parser.parse_args()

    # 尝试打开并读取YAML文件
    with open(args.config_yaml, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            
        except yaml.YAMLError as exc:
            print(exc)  # 打印解析错误

    #expr
    np.random.seed(1234)
    print(config['SET'][0]['Note'])

    if config['SET'][0]['Task'] == "deepxde":
        print("Poisson")
        solver= PDE_PossionData(mu=config['SET'][0]['mu'],shape=config['SET'][0]['shape'])
    else:
       assert False, "PDE name not found"
    

    expr = Expr_Agent(
        solver = solver,
        config = config,
        save_folder = args.save_folder,
        yaml_path = args.config_yaml
    )

    expr.Do_Expr()


