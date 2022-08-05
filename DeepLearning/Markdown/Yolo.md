## Yolov1

+ 特点：
    1. one-stage的目标检测算法
    2. 将目标检测问题看做一个回归问题（将box的分类以及box的定位都转化为回归损失，区别对待有目标和无目标cell的损失权重）
+ 缺点：
    1. 一个cell只能预测一个目标，对多目标情况不友好
    2. 对小目标不友好
    3. 对特殊高宽比的目标检测效果不好
    4. 对小box和大box的误差不应该同等看待

## Yolov2

+ 特点：
    1. 使用新的骨架网络
    2. 为所有卷积层添加BN，取代dropout
    3. 引入anchor机制，每个网格对不同大小的anchor进行预测
    4. 不使用手选的anchor尺寸，而是使用kmeans聚类得到的anchor
    5. RPN网络中预测的box可以为anchor的任意偏置，造成训练不易收敛，yolov2仍保持使用yolov1中预测相对于cell左上角的偏移量来确定box的位置
    6. 为了更好的检测小目标，使用了一个skip-connection,将低层细粒度的26x26的特征图concat到具有更高语义信息的13x13的特征图上
    7. 训练时，每隔10个epoch变换一次图片输入尺寸，使得网络拥有较好的泛化性

## Yolov3

+ 特点：
    1. 骨架网络升级（更深，更多skip-connection）
    2. 参考已有研究，对yolov2做一些工程实现上的升级
    3. 验证了一些无效的改进措施（发现focal loss对yolov3不能带来提升）
    4. 借鉴FPN使用了downsample ratio小的feature map来预测小目标，使用downsample ratio大的feature map预测大目标
    5. 增加了anchor的数量，大目标对应于大尺寸的anchor小目标使用小尺寸的anchor
    6. 之前每个cell可以预测多个box但只能预测一个类别，yolov3则改为每个cell对应的每个anchor都可以预测一个box和一个类别
