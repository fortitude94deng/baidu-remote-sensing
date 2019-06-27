import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#对训练得到的模型进行融合
from Dataloader.MultiModal_BDXJTU2019 import BDXJTU2019_test
from basenet.ResNeXt101_64x4d import ResNeXt101_64x4d
from basenet.senet import se_resnet50,se_resnext101_32x4d
from basenet.octave_resnet import octave_resnet50
from basenet.nasnet import nasnetalarge
from basenet.multimodal import MultiModalNet
from basenet.multimodal1 import MultiModalNet1
from basenet.multimodal2 import MultiModalNet2
import os
# Network
cudnn.benchmark = True 
CLASSES = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def GeResult():

    # Dataset
    Dataset = BDXJTU2019_test(root = '/home/dell/Desktop/2019BaiduXJTU/data')
    Dataloader = data.DataLoader(Dataset, 1,
                                 num_workers = 1,
                                 shuffle = False, pin_memory = True)
    net1 = MultiModalNet1('se_resnet50', 'DPN26', 0.5)
    net1.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet50_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_9.pth'))
    net1.to(device)
    net1.eval()
    net2 = MultiModalNet('se_resnet152', 'DPN26', 0.5)
    net2.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet152_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_6.pth'))
    net2.to(device)
    net2.eval()
    net3 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net3.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_3.pth'))
    net3.to(device) 
    net3.eval()
    net4 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net4.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_10.pth'))
    net4.to(device) 
    net4.eval()   
    net5 = MultiModalNet1('multiscale_se_resnext', 'DPN26', 0.5)
    net5.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/multiscale_se_resnext_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_11.pth'))
    net5.to(device) 
    net5.eval()  
    net6 = MultiModalNet1('multiscale_resnet', 'DPN26', 0.5)
    net6.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/multiscale_resnet_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_10.pth'))
    net6.to(device) 
    net6.eval() 
    net7 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net7.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_4.pth'))
    net7.to(device) 
    net7.eval()    
    #Network = pnasnet5large(6, None)
    #Network = ResNeXt101_64x4d(6)
#    net1 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net1.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained/BDXJTU2019_SGD_16.pth'))
#    net1.eval()

#    net2 = MultiModalNet('multiscale_se_resnext_HR', 'DPN26', 0.5)
#    net2.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_50_MS_resample_pretrained_HR/BDXJTU2019_SGD_26.pth'))
#    net2.eval()


#    net3 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net3.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_w/BDXJTU2019_SGD_50.pth'))
#    net3.eval()


#    net4 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net4.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_1/BDXJTU2019_SGD_80.pth'))
#    net4.eval()

    filename = 'MM_ensemble2.txt'

    f = open(filename, 'w')

    for (Input_O, Input_H, visit_tensor, anos) in Dataloader:
        ConfTensor_O = net1.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_H = net2.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_V = net3.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_1 = net4.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_2 = net5.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_3 = net6.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_4 = net7.forward(Input_O.to(device), visit_tensor.to(device))
        preds = torch.nn.functional.normalize(ConfTensor_O) + torch.nn.functional.normalize(ConfTensor_H) +2*torch.nn.functional.normalize(ConfTensor_V) +torch.nn.functional.normalize(ConfTensor_1)+2*torch.nn.functional.normalize(ConfTensor_2)+torch.nn.functional.normalize(ConfTensor_3)+2*torch.nn.functional.normalize(ConfTensor_4)
        _, pred = preds.data.topk(1, 1, True, True)
        #f.write(anos[0] + ',' + CLASSES[4] + '\r\n')
        print(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
        f.writelines(anos[0][:-4] + '\t' + CLASSES[pred[0][0]] + '\n')
    f.close()
if __name__ == '__main__':
    GeResult()
'''
    net0 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net0.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_3.pth'))
    net0.to(device) 
    net1 = MultiModalNet1('se_resnet50', 'DPN26', 0.5)
    net1.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet50_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_9.pth'))
    net1.to(device)
    net1.eval()
    net2 = MultiModalNet('se_resnet152', 'DPN26', 0.5)
    net2.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/se_resnet152_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_6.pth'))
    net2.to(device)
    net2.eval()
    net3 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net3.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/BDXJTU2019_SGD_1.pth'))
    net3.to(device) 
    net3.eval()
    net4 = MultiModalNet2('densenet201', 'DPN26', 0.5)
    net4.load_state_dict(torch.load('/home/dell/Desktop/2019BaiduXJTU/weights/densenet201_se_resnext50_32x4d_resample_pretrained_80w_1/inception_008.pth'))
    net4.to(device) 
    net4.eval()   
    #Network = pnasnet5large(6, None)
    #Network = ResNeXt101_64x4d(6)
#    net1 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net1.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained/BDXJTU2019_SGD_16.pth'))
#    net1.eval()

#    net2 = MultiModalNet('multiscale_se_resnext_HR', 'DPN26', 0.5)
#    net2.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_50_MS_resample_pretrained_HR/BDXJTU2019_SGD_26.pth'))
#    net2.eval()


#    net3 = MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net3.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_w/BDXJTU2019_SGD_50.pth'))
#    net3.eval()


#    net4 =MultiModalNet('se_resnext50_32x4d', 'DPN26', 0.5)
#    net4.load_state_dict(torch.load('/home/zxw/2019BaiduXJTU/weights/MultiModal_se_resnext50_32x4d_resample_pretrained_1/BDXJTU2019_SGD_80.pth'))
#    net4.eval()

    filename = 'MM_ensemble.txt'

    f = open(filename, 'w')

    for (Input_O, Input_H, visit_tensor, anos) in Dataloader:
        ConfTensor_2 = net0.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_O = net1.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_H = net2.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_V = net3.forward(Input_O.to(device), visit_tensor.to(device))
        ConfTensor_1 = net4.forward(Input_O.to(device), visit_tensor.to(device))
        preds = torch.nn.functional.normalize(ConfTensor_2)+torch.nn.functional.normalize(ConfTensor_O) + torch.nn.functional.normalize(ConfTensor_H) +torch.nn.functional.normalize(ConfTensor_V) +torch.nn.functional.normalize(ConfTensor_1)


'''



























        
