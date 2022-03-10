# -*- coding: UTF-8 -*-
'''
@Project ：DANN_PyTorch
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import torchvision
import torch.nn.functional as F
import config as cfg
from torch import nn
from torch.utils.data import DataLoader
from Dann import DANN

if __name__ == '__main__':

    svhn_dataset = torchvision.datasets.SVHN(root=cfg.fake_sources_path,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
    mnist_train_dataset = torchvision.datasets.MNIST(root=cfg.real_sources_path+'\\train', train=True,
                                                     transform=torchvision.transforms.ToTensor(),
                                                     download=True)
    mnist_test_dataset = torchvision.datasets.MNIST(root=cfg.real_sources_path+'\\test', train=False,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=True)

    svhn_loader = DataLoader(dataset=svhn_dataset, batch_size=cfg.batch_size, shuffle=True)
    mnist_train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=cfg.batch_size, shuffle=True)
    mnist_test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=cfg.batch_size, shuffle=True)

    dann = DANN(device=cfg.device,
                epsilon=cfg.epsilon,
                weight_decay=cfg.weight_decay,
                learning_rate=cfg.learning_rate,
                resume_train=cfg.resume_train,
                ckpt_path=cfg.ckpt_path + "\\模型文件")

    for epoch in range(cfg.Epoches):
        for batch, (mnist_data, svhn_data) in enumerate(zip(list(mnist_train_loader)[:cfg.train_batch_num],
                                                            list(svhn_loader)[:cfg.train_batch_num])):
            real_sources = nn.ReflectionPad2d(padding=2)(mnist_data[0].repeat(1, 3, 1, 1))
            fake_sources = svhn_data[0]
            labels = F.one_hot(mnist_data[1], cfg.class_num).float()

            dann.train(real_sources, fake_sources, labels)

        torch.save({'fe_state_dict': dann.feature_extractor.state_dict(),
                    'lp_state_dict': dann.label_predictor.state_dict(),
                    'dc_state_dict': dann.domain_classifier.state_dict(),
                    'fe_loss': dann.train_fe_loss/(batch+1),
                    'dc_loss': dann.train_dc_loss/(batch+1),
                    'dc_acc': dann.train_dc_acc/(batch+1),
                    'lp_loss': dann.train_lp_loss/(batch+1),
                    'lp_acc': dann.train_lp_acc/(batch+1),
                    'f1_score': dann.f1_score/(batch+1)},
                   cfg.ckpt_path + '\\Epoch{:0>3d}_dc_acc{:.2f}_lp_acc{:.2f}.pth.tar'.format(
                       epoch + 1, dann.train_dc_acc/(batch+1), dann.train_lp_acc/(batch+1)
                   ))

        print(
            f'Epoch {epoch + 1}\n'
            f'fe_Loss: {dann.train_fe_loss/(batch+1)}\n'
            f'dc_Loss: {dann.train_dc_loss/(batch+1)}\n'
            f'dc_f1_score: {dann.f1_score/(batch+1)}\n'
            f'dc_Accuracy: {dann.train_dc_acc/(batch+1) * 100}\n'
            f'lp_Loss:  {dann.train_lp_loss/(batch+1)}\n'
            f'lp_Accuracy: {dann.train_lp_acc/(batch+1) * 100}\n'
        )
        dann.train_fe_loss = 0
        dann.train_dc_loss = 0
        dann.train_dc_acc = 0
        dann.train_lp_loss = 0
        dann.train_lp_acc = 0
        dann.f1_score = 0

        for batch, (mnist_data, svhn_data) in enumerate(zip(list(mnist_test_loader)[:cfg.test_batch_num],
                                                            list(svhn_loader)[cfg.train_batch_num:cfg.train_batch_num+cfg.test_batch_num])):
            real_sources = nn.ReflectionPad2d(padding=2)(mnist_data[0].repeat(1, 3, 1, 1))
            fake_sources = svhn_data[0]
            labels = F.one_hot(mnist_data[1], cfg.class_num).float()

            dann.test(real_sources, fake_sources, labels)

        print(
            f'Epoch {epoch + 1}\n'
            f'test dc_Loss: {dann.test_dc_loss/(batch+1)}\n'
            f'test dc_f1_score: {dann.f1_score/(batch+1)}\n'
            f'test dc_Accuracy: {dann.test_dc_acc/(batch+1) * 100}\n'
            f'test lp_Loss:  {dann.test_lp_loss/(batch+1)}\n'
            f'test lp_Accuracy: {dann.test_lp_acc/(batch+1) * 100}\n'
        )
        dann.test_dc_loss = 0
        dann.test_dc_acc = 0
        dann.test_lp_loss = 0
        dann.test_lp_acc = 0
        dann.f1_score = 0
