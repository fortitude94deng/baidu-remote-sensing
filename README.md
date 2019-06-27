# baidu-remote-sensing
百度遥感比赛，前百分之二的实验方案
multimodal_train_val.py	multimodal_train_val1.py  multimodal_train_val2.py	multimodal_train_val3.py分别对多个模型进行训练，其中有使用单GPU和多GPU
MM_val.py 	MM_val1.py 	MM_val2.py 分别对模型进行单模的验证，并用confusion matrix可视化数据，观察每个模型数据的学习情况，对模型进行筛选，为下一步集成做准备
GeResult_ensemble.py 对模型进行集成，最多使用了七个模型   	GeResult_ensemble_TTA.py，使用TTA集成方法对模型进行集成 
dataloader文件夹中是数据集的制作
