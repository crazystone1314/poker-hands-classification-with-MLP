# poker-hands-classification-with-MLP
poker hands  classification  with  MLP


## 1、所用的到内置库
Numpy
Pandas
Sklearn
Imblearn

## 2、文件说明

neural_network.py      :
mlp_model.pkl          :拥有最高准确率的模型
data_handlers.py       :数据处理
training.py            :训练数据寻找最好的模型
prediction.py          :运用最好的模型进行预测
dataset/               :包括训练集、测试集和预测的结果文件


## 3、神经网络配置

Type                       : 多层感知器 MLPClassfier
Input Layer                : 34 Neuraons
Hidden Layer 1             : 50 Neuraons
Hidden Layer 2             : 50 Neuraons
Hidden layer 3             : 15 Neuraons
Output Layer               : 10 Neuraons
Learning Rate              : 0.001
Momentum Alpha             : 0.0001
Regularizer Lambda         : 0.0001
Loss Function              :
Optimizer                  :
Accuracy                   :

## 4、总结
测试集标签严重不平衡，用分类很难上分的，试过最高才95%左右，都没有比交1高，但经过不均衡处理后，准确率能达到99.8左右！
