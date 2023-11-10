import argparse
from read_data import *
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import time

from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from transformers import ViTModel, ViTConfig



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TH')

    parser.add_argument('--IMG_DIR',default='../TH_scene_source',help='The image directory')
    parser.add_argument('--IS_AUG',default='1',help='Decide whether to use data augmentation or not')
    parser.add_argument('--MODEL',default='VGG16',help='The classification model')

    args = parser.parse_args()

    root_dir = args.IMG_DIR
    is_aug = int(args.IS_AUG)
    model_name = args.MODEL

    epoch = 150

    # Load data without data augmentation
    hazard_scene1 = ExpDataset(root_dir, "hazard", "scene1/site1") + \
            ExpDataset(root_dir, "hazard", "scene1/site2") + \
            ExpDataset(root_dir, "hazard", "scene1/site6") + \
            ExpDataset(root_dir, "hazard", "scene1/site7") + \
            ExpDataset(root_dir, "hazard", "scene1/site8")

    nohazard_scene1 = ExpDataset(root_dir, "nohazard", "scene1/site1") + \
                    ExpDataset(root_dir, "nohazard", "scene1/site2") + \
                    ExpDataset(root_dir, "nohazard", "scene1/site6") + \
                    ExpDataset(root_dir, "nohazard", "scene1/site7") + \
                    ExpDataset(root_dir, "nohazard", "scene1/site8")

    hazard_scene2 = ExpDataset(root_dir, "hazard", "scene2/site3") + \
                    ExpDataset(root_dir, "hazard", "scene2/site4") + \
                    ExpDataset(root_dir, "hazard", "scene2/site5")

    nohazard_scene2 = ExpDataset(root_dir, "nohazard", "scene2/site3") + \
                    ExpDataset(root_dir, "nohazard", "scene2/site4") + \
                    ExpDataset(root_dir, "nohazard", "scene2/site5")

    hazard_scene3 = ExpDataset(root_dir, "hazard", "scene3/site9") + \
                    ExpDataset(root_dir, "hazard", "scene3/site10") + \
                    ExpDataset(root_dir, "hazard", "scene3/site11") + \
                    ExpDataset(root_dir, "hazard", "scene3/site12") + \
                    ExpDataset(root_dir, "hazard", "scene3/site13") + \
                    ExpDataset(root_dir, "hazard", "scene3/site21") 

    nohazard_scene3 = ExpDataset(root_dir, "nohazard", "scene3/site9") + \
                    ExpDataset(root_dir, "nohazard", "scene3/site10") + \
                    ExpDataset(root_dir, "nohazard", "scene3/site11") + \
                    ExpDataset(root_dir, "nohazard", "scene3/site12") + \
                    ExpDataset(root_dir, "nohazard", "scene3/site13") + \
                    ExpDataset(root_dir, "nohazard", "scene3/site21")

    hazard_scene4 = ExpDataset(root_dir, "hazard", "scene4/site14") + \
                    ExpDataset(root_dir, "hazard", "scene4/site15") + \
                    ExpDataset(root_dir, "hazard", "scene4/site16")

    nohazard_scene4 = ExpDataset(root_dir, "nohazard", "scene4/site14") + \
                    ExpDataset(root_dir, "nohazard", "scene4/site15") + \
                    ExpDataset(root_dir, "nohazard", "scene4/site16")

    hazard_scene5 = ExpDataset(root_dir, "hazard", "scene5/site17") + \
        ExpDataset(root_dir, "hazard", "scene5/site18")

    nohazard_scene5 = ExpDataset(root_dir, "nohazard", "scene5/site17") + \
        ExpDataset(root_dir, "nohazard", "scene5/site18")

    hazard_scene6 = ExpDataset(root_dir, "hazard", "scene6/site19") + \
                    ExpDataset(root_dir, "hazard", "scene6/site20")

    nohazard_scene6 = ExpDataset(root_dir, "nohazard", "scene6/site19") + \
                    ExpDataset(root_dir, "nohazard", "scene6/site20")

    hazard_scene7 = ExpDataset(root_dir, "hazard", "scene7/site22") + \
        ExpDataset(root_dir, "hazard", "scene7/site23") + \
        ExpDataset(root_dir, "hazard", "scene7/site24")

    nohazard_scene7 = ExpDataset(root_dir, "nohazard", "scene7/site22") + \
        ExpDataset(root_dir, "nohazard", "scene7/site23") + \
        ExpDataset(root_dir, "nohazard", "scene7/site24")


    hazard_set = [hazard_scene1, hazard_scene2, hazard_scene3, hazard_scene4, hazard_scene5, hazard_scene6, hazard_scene7]
    nohazard_set = [nohazard_scene1, nohazard_scene2, nohazard_scene3, nohazard_scene4, nohazard_scene5, nohazard_scene6, nohazard_scene7]

    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    TN_sum = 0
    for i in range(4):
        # if i==0 or i==1 or i==2 or i==3 or i==4 or i==5:
        #     continue

        if model_name == 'VGG16':
            model = torchvision.models.vgg16(pretrained=True)
            num_fc = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_fc, 2)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        elif model_name == 'EfficientNet':
            model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
            num_classifier = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_classifier, 2)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[1].parameters():
                param.requires_grad = True
        elif model_name == 'ResNet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            num_fc = model.fc.in_features
            model.fc = torch.nn.Linear(num_fc, 2)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'ResNet101':
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            num_fc = model.fc.in_features
            model.fc = torch.nn.Linear(num_fc, 2)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        elif model_name == 'SwinTransformer':
            model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
            num_classifier = model.head.in_features
            model.head = torch.nn.Linear(num_classifier, 2)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            class ViT(nn.Module):
                def __init__(self, config=ViTConfig(), num_labels=2,
                        model_checkpoint='google/vit-base-patch16-224-in21k'):
                    super(ViT, self).__init__()
                    self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
                    self.classifier = (
                        nn.Linear(config.hidden_size, num_labels)
                    )

                def forward(self, x):
                    x = self.vit(x)['last_hidden_state']
                    # Use the embedding of [CLS] token
                    output = self.classifier(x[:, 0, :])
                    return output
                
            model = ViT()

        model = model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(device)

        learning_rate = 0.00003
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if is_aug:
            print("Data augmentation starts...")
            if i == 0:
                hazard_scene1 = ExpDataset(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset(root_dir, "nohazard", "scene1/site8")
                
                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")
            elif i == 1:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")
                
                hazard_scene2 = ExpDataset(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")              
            elif i == 2:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")

                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")
                
                hazard_scene3 = ExpDataset(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")               
            elif i == 3:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")

                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")
                
                hazard_scene4 = ExpDataset(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")               
            elif i == 4:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")

                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")
                
                hazard_scene5 = ExpDataset(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")
            elif i == 5:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")

                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")
                
                hazard_scene6 = ExpDataset(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset(root_dir, "nohazard", "scene6/site20")

                hazard_scene7 = ExpDataset_aug(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset_aug(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene7/site24")
            elif i == 6:
                hazard_scene1 = ExpDataset_aug(root_dir, "hazard", "scene1/site1") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site2") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site6") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site7") + \
                        ExpDataset_aug(root_dir, "hazard", "scene1/site8")

                nohazard_scene1 = ExpDataset_aug(root_dir, "nohazard", "scene1/site1") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site2") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site6") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site7") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene1/site8")

                hazard_scene2 = ExpDataset_aug(root_dir, "hazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "hazard", "scene2/site5")

                nohazard_scene2 = ExpDataset_aug(root_dir, "nohazard", "scene2/site3") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site4") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene2/site5")

                hazard_scene3 = ExpDataset_aug(root_dir, "hazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "hazard", "scene3/site21") 

                nohazard_scene3 = ExpDataset_aug(root_dir, "nohazard", "scene3/site9") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site10") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site11") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site12") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site13") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene3/site21")

                hazard_scene4 = ExpDataset_aug(root_dir, "hazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "hazard", "scene4/site16")

                nohazard_scene4 = ExpDataset_aug(root_dir, "nohazard", "scene4/site14") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site15") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene4/site16")

                hazard_scene5 = ExpDataset_aug(root_dir, "hazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "hazard", "scene5/site18")

                nohazard_scene5 = ExpDataset_aug(root_dir, "nohazard", "scene5/site17") + \
                    ExpDataset_aug(root_dir, "nohazard", "scene5/site18")

                hazard_scene6 = ExpDataset_aug(root_dir, "hazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "hazard", "scene6/site20")

                nohazard_scene6 = ExpDataset_aug(root_dir, "nohazard", "scene6/site19") + \
                                ExpDataset_aug(root_dir, "nohazard", "scene6/site20")
                
                hazard_scene7 = ExpDataset(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset(root_dir, "hazard", "scene7/site24")

                nohazard_scene7 = ExpDataset(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset(root_dir, "nohazard", "scene7/site24")
                
            hazard_set = [hazard_scene1, hazard_scene2, hazard_scene3, hazard_scene4, hazard_scene5, hazard_scene6, hazard_scene7]
            nohazard_set = [nohazard_scene1, nohazard_scene2, nohazard_scene3, nohazard_scene4, nohazard_scene5, nohazard_scene6, nohazard_scene7]

        testset_number = [i]
        trainset_number = [num for num in range(7) if num not in testset_number]

        train_set = hazard_set[trainset_number[0]] + hazard_set[trainset_number[1]] + hazard_set[trainset_number[2]] + \
            hazard_set[trainset_number[3]] + hazard_set[trainset_number[4]] + hazard_set[trainset_number[5]] + \
                nohazard_set[trainset_number[0]] + nohazard_set[trainset_number[1]] + nohazard_set[trainset_number[2]] + \
                    nohazard_set[trainset_number[3]] + nohazard_set[trainset_number[4]] + nohazard_set[trainset_number[5]]

        test_set = hazard_set[testset_number[0]] + nohazard_set[testset_number[0]]

        print("Test on scene {}".format(i+1))
        train_set_size = train_set.__len__()
        test_set_size = test_set.__len__()
        print("The length of train set is {}".format(train_set_size))
        print("The length of test set is {}".format(test_set_size))

        train_dataloader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_set, batch_size=32, num_workers=8, shuffle=True, drop_last=False)

        best_cm = []
        best_test_loss = float('inf')

        epoch_train_loss = []
        epoch_test_loss = []

        for j in range(epoch):
            start = time.time()
            model.train()
            total_train_loss = 0
            for data in train_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets) * len(outputs)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = total_train_loss / train_set_size
            epoch_train_loss.append(average_train_loss)

            model.eval()
            total_test_loss = 0
            labels = []
            preds = []
            with torch.no_grad():
                for data in test_dataloader:
                    imgs, targets = data
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = model(imgs)
                    loss = loss_fn(outputs, targets) * len(outputs)
                    total_test_loss += loss.item()

                    outputs_idx = outputs.argmax(1)
                    idxN = len(targets)
                    for idx in range(idxN):
                        labels.append(targets[idx].item())
                        preds.append(outputs_idx[idx].item())

            cm = confusion_matrix(labels, preds)

            average_test_loss = total_test_loss / test_set_size
            epoch_test_loss.append(average_test_loss)

            if average_test_loss < best_test_loss:
                best_cm = cm
                modelname = f"trainedModels/test{i+1}.pth"
                torch.save(model.state_dict(), modelname)

            end = time.time()

            epoch_time = end - start
            print("Time consuming: {}".format(epoch_time))
        
        def smooth_data(data, smoothing_factor=0.9):
            smoothed_data = []
            last_smoothed_value = data[0]
            for value in data:
                smoothed_value = last_smoothed_value * smoothing_factor + value * (1 - smoothing_factor)
                smoothed_data.append(smoothed_value)
                last_smoothed_value = smoothed_value
            return smoothed_data
        
        epoch_train_loss = smooth_data(epoch_train_loss)
        epoch_test_loss = smooth_data(epoch_test_loss)

        x = [num for num in range(1, epoch+1)]
        cubic_interploation_model1 = interp1d(x,epoch_train_loss,kind="cubic")
        cubic_interploation_model2 = interp1d(x,epoch_test_loss,kind="cubic")
        xs = np.linspace(1,epoch,3*epoch)
        epoch_train_loss = cubic_interploation_model1(xs)
        epoch_test_loss = cubic_interploation_model2(xs)

        plt.figure()
        plt.plot(xs, epoch_train_loss, 'b', label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.title('Train Loss for Test (Scene {})'.format(i + 1))
        plt.savefig(f"experimentGraphs/train_loss_scene{i+1}.jpg")
        plt.close()

        plt.figure()
        plt.plot(xs, epoch_test_loss, 'b', label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.title('Test Loss for Test (Scene {})'.format(i + 1))
        plt.savefig(f"experimentGraphs/test_loss_scene{i+1}.jpg")
        plt.close()

        print("Best confusion matrix is {}".format(best_cm))

        TP = best_cm[0][0]
        FN = best_cm[0][1]
        FP = best_cm[1][0]
        TN = best_cm[1][1]

        TP_sum += TP
        FN_sum += FN
        FP_sum += FP
        TN_sum += TN

        noh_recall = float(float(TP) / (TP+FN))
        noh_precision = float(float(TP) / (TP+FP))

        h_recall = float(float(TN) / (TN+FP))
        h_precision = float(float(TN) / (TN+FN))

        noh_f1_score = float(2 * noh_recall * noh_precision / (noh_recall + noh_precision))
        h_f1_score = float(2*h_recall*h_precision / (h_recall+h_precision))

        noh = (float(TP+FN) / (TP+TN+FN+FP))
        h = 1 - noh

        wF1Score = noh * noh_f1_score + h * h_f1_score
        noh_acc = (float(TP) / (TP+FN))
        h_acc = (float(TN) / (TN+FP))

        print("Best weighted F1-score is {}".format(wF1Score))
        print("Best NoHazard Accuracy is {}".format(noh_acc))
        print("Best Hazard Accuracy is {}".format(h_acc))
    
    print("Experiment Report:")
    noh_recall_all = float(float(TP_sum) / (TP_sum+FN_sum))
    noh_precision_all = float(float(TP_sum) / (TP_sum+FP_sum))

    h_recall_all = float(float(TN_sum) / (TN_sum+FP_sum))
    h_precision_all = float(float(TN_sum) / (TN_sum+FN_sum))

    noh_f1_score_all = float(2 * noh_recall_all * noh_precision_all / (noh_recall_all + noh_precision_all))
    h_f1_score_all = float(2*h_recall_all*h_precision_all / (h_recall_all+h_precision_all))

    noh_all = (float(TP_sum+FN_sum) / (TP_sum+TN_sum+FN_sum+FP_sum))
    h_all = 1 - noh_all

    wF1Score_all = noh_all * noh_f1_score_all + h_all * h_f1_score_all
    noh_acc_all = (float(TP_sum) / (TP_sum+FN_sum))
    h_acc_all = (float(TN_sum) / (TN_sum+FP_sum))
    print("Best weighted F1-score on the whole dataset is {}".format(wF1Score_all))
    print("Best NoHazard Accuracy on the whole dataset is {}".format(noh_acc_all))
    print("Best Hazard Accuracy on the whole dataset is {}".format(h_acc_all))