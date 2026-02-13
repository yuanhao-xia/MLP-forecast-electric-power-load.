# MLP-forecast-electric-power-load.
使用MLP进行电力负荷数值的未来预测，输出为24个点的预测值，基于CNN-LSTM预测代码的修改而来，改变了模型和训练参数的设置，实测的预测效果还不错。
在代码的最开始可以导入自己的负荷数据，或者直接使用生成的具有一定随机性的负荷数据
<img width="4144" height="2968" alt="image" src="https://github.com/user-attachments/assets/5e0ddece-02ab-45b4-8cc6-03a3f36b499d" />
以上是我进行预测的负荷数据，其实这组数据具有很明显的季节性，但是我并没有引入日、周之外的时间特征，后续优化可以引入更多特征让预测更加精确。
预测结果还是可以的，但是对于极值的拟合不是很好，应该是模型本身不会为了追求极限值的精准而放弃周边点的准确性，其实训练参数还是偏于保守，还可以更加激进，但对于我的目标来说是足够的。
<img width="4469" height="1764" alt="image" src="https://github.com/user-attachments/assets/0ab409e0-6b2e-4611-bed0-4cd65b98bd89" />

