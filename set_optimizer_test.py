import torch.nn as nn
import torch
from nets.yolo_training import weights_init
from nets.yolo import YoloBody
from utils.utils import get_anchors, get_classes
import numpy as np
import torch.optim as optim


classes_path    = 'model_data/voc_classes.txt'
anchors_path    = 'model_data/yolo_anchors.txt'
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
model_path      = 'model_data/yolo4_weights.pth'



class_names, num_classes = get_classes(classes_path)  # 列表，数
anchors, num_anchors = get_anchors(anchors_path)  # 矩阵，数

# ------------------------------------------------------#
#   创建yolo模型
# ------------------------------------------------------#
model = YoloBody(anchors_mask, num_classes, pretrained=False)  # pretrained用于判断创建yolov4网络时是否加载主干特征网络的预训练权重。
if not False:  # 如果没加载主干网络的特征那么初始化网络权重
    weights_init(model)
if model_path != '':  # 加载整个网络的权重
    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    # print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()  # 将模型里面的参数（以字典形式存在）赋给model_dict
    pretrained_dict = torch.load(model_path, map_location=device)  # 加载预训练模型,参数赋给pretrained_dict

    # 将 pretrained_dict 里不属于 model_dict 的键剔除掉。比如可能修改了一部分网络，那么需要过滤掉这些参数。
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)  # 更新现有的 model_dict
    model.load_state_dict(model_dict)  # 加载我们真正需要的 state_dict



pg0, pg1, pg2 = [], [], []
for k, v in model.named_modules():  # key value
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)
#print(len(pg0),len(pg1),len(pg2))
# print(pg0)
# print(pg1)
# print(pg2)
for k,v in model.named_modules():
    pass
    #print(v)



class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.model =  nn.Sequential(
            nn.Conv2d(3, 10, 3, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            nn.Conv2d(10, 20, 3),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        return self.model(x)
model1 = TestModel()
weights_init(model1)

list1 = []
list2 = []
pg0, pg1, pg2 = [], [], []
for k, v in model.named_modules(): #key value，其中key是层（layers）的名字（name），v是层（layers）
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias) #如果层v含有偏置属性bias，且该偏置属于nn.Parameters类型，即可训练且可注册到网络中的参数，那么就添加到列表pg2中
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight) #如果层v是nn.BatchNorm2d类型，或层的名字里有bn俩字母，那么就将BN层的参数加到pg0中
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
         pg1.append(v.weight) #如果层v含有权重weight，且该偏置属于nn.Parameters类型，即可训练且可注册到网络中的参数，那么就添加到列表pg1中


optimizer = optim.Adam(pg0, 0.01, betas=(0.97, 0.999))
optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4, 'lr':0.001})
optimizer.add_param_group({"params": pg2})

print(optimizer.param_groups)






