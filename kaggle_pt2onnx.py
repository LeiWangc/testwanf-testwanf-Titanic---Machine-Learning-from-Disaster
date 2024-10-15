import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
from gensim.models import KeyedVectors
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import csv
from kaggle_read import model,CSVDataset
import onnxruntime as ort

if __name__ =="__main__":
    Model = model(53).cuda()
    Model.load_state_dict(torch.load('model_weights.pth'))
    input_data = torch.from_numpy(np.random.randn(1, 53).astype(np.float32)).cuda()  # 根据输入大小调整
    torch.onnx.export(Model,                # 要导出的模型
                  input_data,          # 模型输入 (或一组输入)
                  "model.onnx",        # 保存路径
                  export_params=True,   # 是否导出模型参数
                  opset_version=11,     # ONNX 版本
                  do_constant_folding=True,  # 是否做常量折叠
                  input_names=['input'],  # 输入名称
                  output_names=['output'], # 输出名称
                  dynamic_axes={'input': {0: 'batch_size'},  # 动态轴
                                'output': {0: 'batch_size'}})
    # test_file_path = './titanic/test.csv'
    # forw_dataset = CSVDataset(test_file_path)
    # val_data = DataLoader(forw_dataset, batch_size=2, shuffle=False)
    # data ,label ,num = next(iter(val_data))
    # onnx_model_path = "model.onnx"
    # session = ort.InferenceSession(onnx_model_path)
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # input_shape = session.get_inputs()[0].shape
    # print(input_name,output_name,input_shape)
    # outputs = session.run([output_name], {input_name: data.cpu().numpy()})
    # output = (outputs[0]>0.5).astype(int)
    # print(output)
