# testwanf-Titanic---Machine-Learning-from-Disaster
1.直接运行kaggle_train.py,最终会生成符合网站提交格式的csv文件和保存好的模型文件。

2.运行kaggle_pt2onnx.py,将模型转换成onnx格式。

3.在ncnn文件中进行编译安装，利用onnx2ncnn工具把onnx格式数据转换成param和bin后缀的ncnn格式文件

4.在ncnn_deployment文件中进行编译运行，能够直接部署推理
