import yaml

def read_yaml(yaml_path):
    # 读取Yaml文件方法
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result