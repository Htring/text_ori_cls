import os
import ujson as json

file_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(file_dir, "./models")
config_path = os.path.join(file_dir, "./config.json")


with open(config_path, 'r') as reader:
    model_param = json.load(reader)

model_param['model'] = os.path.join(model_dir, model_param['model'])
