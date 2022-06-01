# -*- coding: UTF-8 -*-
'''
@Project ：DANN_PyTorch
@File    ：Dann.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import numpy as np
from torch import nn
from net import Feature_extractor, Domain_classifier, Label_predictor

class DANN:
    def __init__(self,
                 device,
                 epsilon,
                 weight_decay,
                 learning_rate,
                 resume_train,
                 ckpt_path,
                 **kwargs):

        assert np.logical_and(np.equal(len(learning_rate), 3),
                              np.equal(len(weight_decay), 3))

        self.device = device

        self.feature_extractor = Feature_extractor()
        self.domain_classifier = Domain_classifier()
        self.label_predictor = Label_predictor()

        if self.device:
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.domain_classifier = self.domain_classifier.to(self.device)
            self.label_predictor = self.label_predictor.to(self.device)

        if resume_train:
            ckpt = torch.load(ckpt_path)
            fe_ckpt_dict = ckpt['fe_state_dict']
            dc_ckpt_dict = ckpt['dc_state_dict']
            lp_ckpt_dict = ckpt['lp_state_dict']
            self.feature_extractor.load_state_dict(fe_ckpt_dict)
            self.domain_classifier.load_state_dict(dc_ckpt_dict)
            self.label_predictor.load_state_dict(lp_ckpt_dict)

        self.epsilon = epsilon
        self.fe_decay = weight_decay[0]
        self.dc_decay = weight_decay[1]
        self.lp_decay = weight_decay[2]

        self.lp_loss = nn.BCELoss(reduction='mean')

        self.fe_optimizer = torch.optim.Adam(params=self.feature_extractor.parameters(),
                                             lr=learning_rate[0])
        self.dc_optimizer = torch.optim.Adam(params=self.domain_classifier.parameters(),
                                             lr=learning_rate[1])
        self.lp_optimizer = torch.optim.Adam(params=self.label_predictor.parameters(),
                                             lr=learning_rate[2])

        self.train_dc_loss, self.test_dc_loss = 0, 0
        self.train_dc_acc, self.test_dc_acc = 0, 0
        self.train_lp_loss, self.test_lp_loss = 0, 0
        self.train_lp_acc, self.test_lp_acc = 0, 0
        self.train_fe_loss = 0
        self.f1_score = 0

    def dc_loss(self, real, fake):

        real_loss = torch.mean(torch.relu(1. - torch.ones_like(real)*real))
        fake_loss = torch.mean(torch.relu(1. + torch.ones_like(fake)*fake))

        loss = (real_loss + fake_loss)/2.

        return loss

    def train(self, real_sources, fake_sources, labels):
        """
        the order of gradients' calculation is different,
        dc_model&fe_model can not optimize at the same time
        :param real_sources: real samples
        :param fake_sources: fake samples
        :param labels: label of real samples
        """
        if self.device:
            real_sources = real_sources.to(self.device)
            fake_sources = fake_sources.to(self.device)
            labels = labels.to(self.device)

        # domain_classifier
        self.dc_optimizer.zero_grad()

        real_features = self.feature_extractor(real_sources).detach()
        fake_features = self.feature_extractor(fake_sources).detach()
        real_domains = self.domain_classifier(real_features)
        fake_domains = self.domain_classifier(fake_features)
        dc_loss = self.dc_loss(real_domains, fake_domains)
        for param in self.domain_classifier.weights:
            dc_loss += self.dc_decay*torch.sum(torch.square(param))
        dc_loss.backward()
        self.dc_optimizer.step()

        self.train_dc_loss += dc_loss.data.item()
        self.train_dc_acc += (torch.sum(torch.gt(real_domains.squeeze(1), 0)).data.item() +
                              torch.sum(torch.less_equal(fake_domains.squeeze(1), 0)).data.item())\
                             /(labels.size()[0]*2)
        self.calculate_f1_score(real_domains, fake_domains)

        # feature_extractor & label predictor
        self.fe_optimizer.zero_grad()
        self.lp_optimizer.zero_grad()

        real_features = self.feature_extractor(real_sources)
        fake_features = self.feature_extractor(fake_sources)
        real_domains = self.domain_classifier(real_features)
        fake_domains = self.domain_classifier(fake_features)
        dc_loss = self.dc_loss(real_domains, fake_domains)

        features = self.feature_extractor(real_sources)
        predictions = self.label_predictor(features)
        lp_loss = self.lp_loss(predictions, labels)

        extract_loss = lp_loss - dc_loss
        for param in self.feature_extractor.weights:
            extract_loss += self.fe_decay*torch.sum(torch.square(param))
        self.train_fe_loss += extract_loss.data.item()

        for param in self.label_predictor.weights:
            extract_loss += self.lp_decay*torch.sum(torch.square(param))
            lp_loss += self.lp_decay*torch.sum(torch.square(param))
        self.train_lp_loss += lp_loss.data.item()
        self.train_lp_acc += torch.sum((torch.eq(predictions.argmax(dim=-1),
                                                 labels.argmax(dim=-1)))).data.item()/labels.size()[0]

        extract_loss.backward()

        self.lp_optimizer.step()
        self.fe_optimizer.step()

    def test(self, real_sources, fake_sources, labels):
        """
        just calculate lp_model's、dc_model's loss&acc
        :param real_sources: real samples
        :param fake_sources: fake samples
        :param labels: label of real samples
        """
        if self.device:
            real_sources = real_sources.to(self.device)
            fake_sources = fake_sources.to(self.device)
            labels = labels.to(self.device)

        # domain_classifier
        real_features = self.feature_extractor(real_sources)
        fake_features = self.feature_extractor(fake_sources)
        real_domains = self.domain_classifier(real_features)
        fake_domains = self.domain_classifier(fake_features)
        dc_loss = self.dc_loss(real_domains, fake_domains)
        for param in self.domain_classifier.weights:
            dc_loss += self.dc_decay*torch.sum(torch.square(param))

        self.test_dc_loss += dc_loss.data.item()
        self.test_dc_acc += (torch.sum(torch.gt(real_domains.squeeze(1), 0)).data.item() +
                             torch.sum(torch.less_equal(fake_domains.squeeze(1), 0)).data.item())\
                            /(labels.size()[0]*2)
        self.calculate_f1_score(real_domains, fake_domains)

        # label_predictor
        features = self.feature_extractor(real_sources)
        predictions = self.label_predictor(features)
        lp_loss = self.lp_loss(predictions, labels)
        for param in self.label_predictor.weights:
            lp_loss += self.lp_decay*torch.sum(torch.square(param))

        self.test_lp_loss += lp_loss.data.item()
        self.test_lp_acc += torch.sum((torch.eq(predictions.argmax(dim=-1),
                                                labels.argmax(dim=-1)))).data.item()\
                            /labels.size()[0]

    def calculate_f1_score(self, real_domains, fake_domains):
        """
        Calculate only domain classifier's f1,
        using hinge loss, so different from the traditional method
        for the label predictor, the distribution of multi-class samples in a batch are unbalanced
        """

        real_dc_tp = torch.sum(torch.gt(real_domains.squeeze(1), 0)).data.item()
        fake_dc_tp = torch.sum(torch.less_equal(fake_domains.squeeze(1), 0)).data.item()
        real_dc_fp = torch.sum(torch.less_equal(real_domains.squeeze(1), 0)).data.item()
        fake_dc_fp = torch.sum(torch.gt(fake_domains.squeeze(1), 0)).data.item()
        real_dc_recall = real_dc_tp/(real_dc_tp+real_dc_fp)
        real_dc_precision = real_dc_tp/(real_dc_tp+fake_dc_fp+self.epsilon)
        fake_dc_recall = fake_dc_tp/(fake_dc_tp+fake_dc_fp)
        fake_dc_precision = fake_dc_tp/(fake_dc_tp+real_dc_fp+self.epsilon)
        real_dc_f1 = 2*real_dc_precision*real_dc_recall/(real_dc_precision+real_dc_recall+self.epsilon)
        fake_dc_f1 = 2*fake_dc_precision*fake_dc_recall/(fake_dc_precision+fake_dc_recall+self.epsilon)
        self.f1_score += ((real_dc_f1+fake_dc_f1)/2)**2

