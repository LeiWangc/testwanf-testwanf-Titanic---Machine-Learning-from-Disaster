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

#去掉标点符号
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
#去掉空格
def remove_spaces(s):
    if ' ' in s:
        return s.replace(' ', '')
    return s
#输出ascall码值
def convert_letters_to_ascii(s):
    return ''.join(str(ord(char)) if char.isalpha() else char for char in s)
#数字转换向量
def convert_to_fixed_length_vector(num_str, length=25):
    # 将字符串转换为整数列表
    num_list = [int(digit) for digit in num_str]
    # 创建长度为 15 的零向量
    vector = np.zeros(length, dtype=int)
    # 计算填充的起始位置

    start_index = length - len(num_list)
    # print(len(num_list))
    # 将数字填入向量
    vector[start_index:] = num_list
    return vector

#求一个名字额平均向量


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.mode_path = "./GoogleNews-vectors-negative300.bin.gz"
        self.model = KeyedVectors.load_word2vec_format(self.mode_path, binary=True)
        self.my_dict = {
        "C": 1,
        "Q": 2,
        "S": 3
        }
    def sentence_to_vector(self,sentence):
        words = sentence.split()
        vectors = [self.model[word] for word in words if word in self.model]
        return np.mean(vectors, axis=0)  # 计算句子的平均词向量
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        item = self.data_frame.iloc[idx]
        # print(item)
        num = item[0]
        feature =item[-10:].values
        label =item[-11]
        words = remove_punctuation(feature[1])
        feature[1] = 1#self.sentence_to_vector(words)
        feature[2] = 0 if feature[2]=='male' else 1
        # print(feature[3].dtype)
        # print((feature[3] != " "))
        feature[3] = float(feature[3] if (isinstance(feature[3], float) and not np.isnan(feature[3])) else 0)
        feature[-4] = convert_to_fixed_length_vector(convert_letters_to_ascii(remove_spaces(remove_punctuation(feature[-4]))))
        feature[-2] = convert_to_fixed_length_vector(convert_letters_to_ascii(remove_spaces(feature[-2])),length=20) if isinstance(feature[-2], str) and feature[-2] != " " else np.zeros(20)
        
        feature[-1]= self.my_dict[feature[-1]] if isinstance(feature[-1], str) and feature[-1] != " " else 0

        flattened_data = np.concatenate([item.flatten() if isinstance(item, np.ndarray) else np.array([item]) for item in feature])
        flattened_data[3] = flattened_data[3] * 0.1
        features = torch.tensor(flattened_data, dtype=torch.float32)
        
        labels = torch.tensor(label, dtype=torch.float32).cuda()
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        # 标准化操作
        # print(flattened_data)
        features_normalized = ((features - mean) / std).cuda()
        return features_normalized, labels, num



class model(nn.Module):
    def __init__(self,input_size):
        super(model,self).__init__()
        self.lin1 = nn.Linear(input_size,80)
        self.lin2 = nn.Linear(80,160)
        # self.lin21 = nn.Linear(160,320)
        # self.lin22 = nn.Linear(320,160)
        self.lin3 = nn.Linear(160,80)
        self.lin4 = nn.Linear(80,40)
        self.lin5 = nn.Linear(40,20)
        self.lin6 = nn.Linear(20,1)
        self.relu = nn.ReLU()         # 激活函数
        self.sigmoid = nn.Sigmoid()   # 二分类，使用Sigmoid激活函数

        self.Sequentials = nn.Sequential(
            self.lin1,
            self.relu,
            self.lin2,
            # self.relu,
            # self.lin21,
            # self.relu,
            # self.lin22,
            # self.relu,
            self.lin3,
            self.relu,
            self.lin4,
            self.relu,
            self.lin5,
            self.relu,
            self.lin6,
            nn.Dropout(0.5),
            self.sigmoid
        )
    def forward(self,input):
        output  = self.Sequentials(input)
        return output.squeeze(1)

def train_model(model, criterion, optimizer, train_data, test_data, num_epochs=10000):
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"): 
        for features, labels,_ in train_data:
            model.train()
            optimizer.zero_grad()
            length = features.size(1)
            outputs = model(features)
            #print(outputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        #每10个epoch打印一次损失
        model.eval()
        if (epoch+1) % 10 == 0:  
            with torch.no_grad():   
                for features_t, labels_t,_ in test_data:
                    outputs_t = model(features_t)
                    val_loss = criterion(outputs_t, labels_t)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
def val_model(model, val_data, writer):
    model.eval()
    with torch.no_grad():   
        for features_v, labels_v , num in val_data:
            outputs_v = model(features_v)
            output = (outputs_v>0.5).int().cpu()
            tensor_num = output.numel()
            for i in range(tensor_num):
                writer.writerow([num[i].item(),output[i].item()])
            print(output)
if __name__ =="__main__":
    data = [
    ['PassengerId','Survived']
    ]
    file = open('output.csv', mode='w', newline='')
    writer = csv.writer(file)
    writer.writerows(data)
    file = open('output.csv', mode='a', newline='')
    writer = csv.writer(file)

    csv_file_path = './titanic/train.csv'
    test_file_path = './titanic/test.csv'
    my_dataset = CSVDataset(csv_file_path)
    forw_dataset = CSVDataset(test_file_path)
    # 定义训练集和测试集的比例
    train_size = int(0.8 * len(my_dataset))  # 80%的数据用于训练
    test_size = len(my_dataset) - train_size  # 剩余的数据用于测试

    # 使用 random_split 划分数据集
    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])
    train_data = DataLoader(my_dataset, batch_size=16, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=16, shuffle=True)
    val_data = DataLoader(forw_dataset, batch_size=16, shuffle=False)


    Model = model(53).cuda()
    criterion = nn.BCELoss()  # 二分类损失函数，使用BCELoss
    optimizer = optim.Adam(Model.parameters(), lr=0.0001)

    train_model(Model,criterion,optimizer,train_data,test_data)
    torch.save(Model.state_dict(), 'model_weights.pth')
    # Model.load_state_dict(torch.load('model_weights.pth'))
    val_model(Model,val_data,writer)
    #for features, labels in dataloader:
       #print(features.shape)
       #break

    # mode_path = "./GoogleNews-vectors-negative300.bin.gz"
    # model = KeyedVectors.load_word2vec_format(mode_path, binary=True)
    # my_dict = {
    # "C": 1,
    # "Q": 2,
    # "S": 3
    # }
    # df = pd.read_csv(csv_file_path)
    # feature = df.iloc[0][2:].values
    # words = remove_punctuation(feature[1])
    # feature[1] = sentence_to_vector(words)
    # feature[2] = 0 if feature[2]=='male' else 1
    # feature[3] = int(feature[3]) if feature[3].dtype == np.float64 else 0
    # # feature[-2] = model[feature[-2]] if feature[-2].dtype==np.float64 else np.zeros(300)
    # feature[-4] = convert_to_fixed_length_vector(convert_letters_to_ascii(remove_spaces(remove_punctuation(feature[-4]))))
    # feature[-2] = model[feature[-2]] if isinstance(feature[-2], str) and feature[-2] != " " else np.zeros(300)
    # feature[-1]= my_dict[feature[-1]]

    # flattened_data = np.concatenate([item.flatten() if isinstance(item, np.ndarray) else np.array([item]) for item in feature])
    # print(flattened_data.shape)
