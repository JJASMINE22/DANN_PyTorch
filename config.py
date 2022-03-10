# -*- coding: UTF-8 -*-
'''
@Project ：DANN_pytorch
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch

# ===data_loader===
real_sources_path = 'C:\\DATASET\\mnist'
fake_sources_path = 'C:\\DATASET\\svhn'
train_batch_num = 1800
test_batch_num = 300
class_num = 10

# ===training===
Epoches = 100
batch_size = 32
epsilon = 1e-7
weight_decay = [0.0001, 0.0002, 0.00005]
learning_rate = [0.005, 0.0001, 0.001]
device = torch.device('cuda') if torch.cuda.is_available() else None
ckpt_path = '.\\saved'
resume_train = False

