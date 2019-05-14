# FER-based-on-Pytorch

本仓库存储自己毕设所写的基于pytorch框架的人脸表情识别系统，代码目录情况在本read_me文件最下方

依赖
----
windows 10    
cuda：10.1.105（cuda_10.1.105_win10_network.exe）    
torch: 1.0.1
face_recognition框架
numpy, pandas, seaborn, matplotlib 等，报什么缺失补什么

superconfig.sh 中的代码是从零搭建所需环境（申请了阿里云的环境搭建过程）

本仓库的代码介绍
----
其中，.ipynb后缀文件是用于测试代码块，对本项目的完整性无影响。可以作为学习查看，也可以忽略。
1. pre_processing_data文件夹内是对数据集进行预处理，并生成data文件要求的目录结构，以及对face_recognition进行学习和代码测试
2. dal文件夹是对CK+数据集、JAFFE数据集和FER2013数据集进行的封装
3. networks文件夹是对网络进行的封装（包含自己搭建的简单ACNN网络（前期测试训练和测试的代码块）、自己搭建的ACCNN网络（后期自行提出的新模型）、AlexNet网络、VGG网络、ResNet网络）
4. transforms是对数据预处理方法的封装（非自己写）
5. utils是对常用的公用函数的封装（自己写的）
6. main_windows是使用PyQt5搭建的人脸表情识别系统界面，以及界面各层：界面-->worker子线程-->model_controller-->模型，子线程完成任务通过信号回传界面

以下文件按照时间先后次序介绍。
1. cnn.ipynb  最开始用于学习pytorch搭建模型创建
2. train_test.py 测试和训练模型的主函数
3. train_test_all_model.py 训练测试所有的模型（调用cmd，执行完一个子程序执行下一个，后期转train_test.ipynb进行模型训练，该文件废弃，未后续更新）
4. virtualize.py  用于模型中间层输出的可视化，对函数进行封装
5. train_test.ipynb  主要的后期训练模型的文件，用于在Google Colab上面利用免费GPU算力进行训练
6. trained_model_results.ipynb  展示模型训练的结果
7. compute_receptive_field.ipynb  计算模型神经元的感受野
8. test_dataset_with_trained_model.ipynb  在数据集上测试训练完的模型

目录
----
其中仓库未提供：data文件夹（需要按照下述data目录进行放置）、Saved_Models文件夹（训练过程中自动生成，存储训练的模型参数和训练进度）、Saved_Virtualizations文件夹（运行代码自动生成，存储可视化图片，以及一些统计图片）    
```
.      
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
...

.
├── data
├── Saved_* 文件夹
├── pre_processing_data
│   ├── re_build_data_CKPlus48.ipynb
│   ├── re_build_data_CKPlus.ipynb
│   ├── re_build_data_face_recognition.ipynb
│   ├── re_build_data_fer2013.ipynb
│   └── re_build_data_JAFFE.ipynb
├── dal
│   ├── CKPlus_DataSet.py
│   ├── FER2013_DataSet.py
│   ├── __init__.py
│   └── JAFFE_DataSet.py
├── networks
│   ├── ACCNN.py
│   ├── ACNN.py
│   ├── AlexNet.py
│   ├── __init__.py
│   ├── ResNet.py
│   └── VGG.py
├── transforms
│   ├── functional.py
│   ├── __init__.py
│   └── transforms.py
├── utils
│   ├── face_recognition.py
│   ├── __init__.py
│   └── utils.py
├── main_windows
│   ├── css.py
│   ├── FER_Window.py
│   ├── __init__.py
│   ├── model_controller.py
│   ├── Resources
│   │   ├── ACCNN_model_structer.png
│   │   ├── Rectangle_Black.png
│   │   └── Rectangle.png
│   └── worker_threads.py
├── __init__.py
├── main.py
├── train_test_all_model.py
├── train_test.py
├── test_dataset_with_trained_model.ipynb
├── trained_model_results.ipynb
├── train_test.ipynb
├── cnn.ipynb
├── compute_receptive_field.ipynb
└── virtualize.py
``` 
