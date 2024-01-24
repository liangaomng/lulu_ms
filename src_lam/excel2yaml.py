import pandas as pd
import yaml
from pathlib import Path
import os

class Excel2yaml():
    def __init__(self,path,excel):

        self.save_ymlpath=path+".yaml" #yml 路径
        self.excel_path=excel
        # 判断保存路径是否存在，不存在则创建
        # 如果路径是一个文件的路径，分离出目录部分
        directory = os.path.dirname(self.save_ymlpath)

        try:
            if not os.path.exists(directory):

                os.makedirs(path,exist_ok=True)
                #创建空的ymal

                print(f"Directory created: {directory}")
            else:
                print(f"Directory already exists: {directory}")
        except Exception as e:
            print(f"Error creating directory: {e}")



    def excel2yaml(self,exhit='LossRecord'):
        # 读取 Excel 文件中的所有 sheets
        xls = pd.ExcelFile(self.excel_path)
        # Get all sheet names and exclude 'LossRecord'
        sheet_names = [sheet_name for sheet_name in xls.sheet_names if sheet_name != exhit]

        # 初始化一个字典来存储所有 sheets 的数据，排除 'LossRecord' 工作表
        sheets_data = {}
        for sheet in sheet_names:
            if sheet != exhit:
                df = pd.read_excel(self.excel_path, sheet_name=sheet)

                # 处理空值：可以选择删除含空值的行，或填充默认值
                df = df.dropna()  # 删除含空值的行

                # 删除重复行
                df = df.drop_duplicates()

                # 转换为字典
                sheets_data[sheet] = df.to_dict(orient='records')
                # 将字典转换为 YAML 格式的字符串

        yaml_data = yaml.dump(sheets_data, allow_unicode=True)


        # 将 YAML 数据保存到文件
        with open(self.save_ymlpath, 'w', encoding='utf-8') as file:
            file.write(yaml_data)


if __name__=="__main__":
    pass