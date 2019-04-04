# FER-based-on-Pytorch

本仓库存储自己毕设所写的基于pytorch框架的人脸表情识别系统

依赖
----
windows 10

cuda：10.1.105（cuda_10.1.105_win10_network.exe）

torch: 1.0.1

numpy, pandas, seaborn, matplotlib 等，报什么缺失补什么


目录
----
其中仓库未提供：data文件夹（需要按照下述目录进行放置）和Saved_Models文件夹（训练过程中自动生成）
E:.
├─.idea
├─dal
├─data
│  ├─CK+
│  │  ├─anger
|  |  |  └─各表情图片
│  │  ├─contempt
|  |  |  └─各表情图片
│  │  ├─disgust
|  |  |  └─各表情图片
│  │  ├─fear
|  |  |  └─各表情图片
│  │  ├─happy
|  |  |  └─各表情图片
│  │  ├─sadness
|  |  |  └─各表情图片
│  │  └─surprise
|  |  |  └─各表情图片
│  ├─CK+48
│  │  ├─anger
|  |  |  └─各表情图片
│  │  ├─contempt
|  |  |  └─各表情图片
│  │  ├─disgust
|  |  |  └─各表情图片
│  │  ├─fear
|  |  |  └─各表情图片
│  │  ├─happy
|  |  |  └─各表情图片
│  │  ├─sadness
|  |  |  └─各表情图片
│  │  └─surprise
|  |     └─各表情图片
│  ├─fer2013
|  |  └─fer2013.csv
│  └─jaffe
│      ├─KA
|      |  └─该人物的表情图片
│      ├─KL
|      |  └─该人物的表情图片
│      ├─KM
|      |  └─该人物的表情图片
│      ├─KR
|      |  └─该人物的表情图片
│      ├─MK
|      |  └─该人物的表情图片
│      ├─NA
|      |  └─该人物的表情图片
│      ├─NM
|      |  └─该人物的表情图片
│      ├─TM
|      |  └─该人物的表情图片
│      ├─UY
|      |  └─该人物的表情图片
│      └─YM
|         └─该人物的表情图片
├─networks
├─pre_processing_data
├─Saved_Models
│  ├─CK+48_ACNN_1
│  └─CK+48_AlexNet_1
├─transforms
├─utils
