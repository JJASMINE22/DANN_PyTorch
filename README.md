## DANN：Domain adversarial neural network模型的PyTorch实现
---

## 目录
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [训练步骤 How2train](#训练步骤) 

## 所需环境
1. Python3.7
2. PyTorch>=1.7.0+cu110	
3. TorchVision>=0.8.1+cu110
4. Numpy==1.19.5	
5. CUDA 11.0+

## 注意事项
1. DANN结构擅于避免模型过学习 
2. feature_extractor与domain_classifier模块合并构成域分类器
3. feature_extractor与label_predictor模块合并构成样本分类器
4. 通过输入真实数据与抽象数据，输出基于域分类的dc_loss，用于domain_classifier的反向传递
5. 将真实数据输入样本分类器，将lp_loss作用于feature_extractor,并将lp_loss-dc_loss作用于label_predictor
6.	优化单个分体模型时，将计算合并模型的梯度，需使用detach()或zero_grad()转为常量

## 训练步骤
1. 默认使用mnist作为真实样本，svhn作为抽象样本
2. 首次运行将自行下载以上两种数据集
3. 运行train.py即可开始训练
