'''
This python file is to test a single network on dataset and visiualise confusion metrix
在验证集上分析训练得到的多模型， 里面的confusion matrix可以分析分类结果

'''
import os
import csv
import h5py
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score

from Dataloader.MultiModal_BDXJTU2019 import MM_BDXJTU2019
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet
from basenet.multimodal1 import MultiModalNet1
from basenet.multimodal2 import MultiModalNet2
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print(np.diag(cm))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def GeResult():

    # Dataset
    Dataset_val = MM_BDXJTU2019(root='/home/dell/Desktop/2019BaiduXJTU/data', mode = 'val')
    Dataloader_val = data.DataLoader(Dataset_val, batch_size = 1,
                                 num_workers = 2,
                                 shuffle = True, pin_memory = True)


    class_names = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
    
    net1 = MultiModalNet1('se_resnet50', 'DPN26', 0.5)
    net1.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet50_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_9.pth'))
    net1.to(device)
    net1.eval()
    net2 = MultiModalNet('se_resnet152', 'DPN26', 0.5)
    net2.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet152_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_4.pth'))
    net2.to(device)
    net2.eval()
    net3 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net3.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_3.pth'))
    net3.to(device) 
    net3.eval()
    # construct network
#    net1 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net1.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/models/BDXJTU2019_SGD_16.pth'))
#    net1.eval()

#    net2 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net2.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/models/BDXJTU2019_SGD_26.pth'))
#    net2.eval()

#    net3 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net3.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/models/BDXJTU2019_SGD_50.pth'))
#    net3.eval()

    results = []
    results_anno = []
    
    for i, (Input_img, Input_vis, Anno) in enumerate(Dataloader_val):
        Input_img = Input_img.to(device)
        Input_vis = Input_vis.to(device)


        ConfTensor1 = net1.forward(Input_img, Input_vis)
        ConfTensor2 = net2.forward(Input_img, Input_vis)
        ConfTensor3 = net3.forward(Input_img, Input_vis)

        ConfTensor = (torch.nn.functional.normalize(ConfTensor1) + torch.nn.functional.normalize(ConfTensor2) +torch.nn.functional.normalize(ConfTensor3))/3
   
        score, pred = ConfTensor.data.topk(1, 1, True, False)
        #print(score.item())
        if(score.item()>0.85) :
            results.append(pred.item())

            results_anno.append(Anno)                                #append annotation results
        if((i +1)%2000 == 0 ):
            print(i+1)
            print(len(results))
            print('Accuracy of Orignal Input: %0.6f'%(accuracy_score(results, results_anno, normalize = True)))
    # print accuracy of different input
    print('Accuracy of Orignal Input: %0.6f'%(accuracy_score(results, results_anno, normalize = True)))

    cnf_matrix = confusion_matrix(results_anno, results)
    cnf_tr = np.trace(cnf_matrix)
    cnf_tr = cnf_tr.astype('float')
    print(cnf_tr/len(Dataset_val))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names ,title='Confusion matrix, without normalization')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    
if __name__ == '__main__':
    GeResult()
 


























       
