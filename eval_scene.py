import argparse
from read_data import *
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

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

    parser.add_argument('--TEST_DIR',default='../TH_scene_source',help='The test image directory')
    parser.add_argument('--SCENE',default='1',help='The test scene number')
    parser.add_argument('--MODEL',default='VGG16',help='The classification model')

    args = parser.parse_args()

    root_dir = args.TEST_DIR
    scene_num = int(args.SCENE)
    model_name = args.MODEL

    if scene_num == 1:
        hazard_test = ExpDataset(root_dir, "hazard", "scene1/site1") + \
                ExpDataset(root_dir, "hazard", "scene1/site2") + \
                ExpDataset(root_dir, "hazard", "scene1/site6") + \
                ExpDataset(root_dir, "hazard", "scene1/site7") + \
                ExpDataset(root_dir, "hazard", "scene1/site8")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene1/site1") + \
                        ExpDataset(root_dir, "nohazard", "scene1/site2") + \
                        ExpDataset(root_dir, "nohazard", "scene1/site6") + \
                        ExpDataset(root_dir, "nohazard", "scene1/site7") + \
                        ExpDataset(root_dir, "nohazard", "scene1/site8")
    elif scene_num == 2:
        hazard_test = ExpDataset(root_dir, "hazard", "scene2/site3") + \
                        ExpDataset(root_dir, "hazard", "scene2/site4") + \
                        ExpDataset(root_dir, "hazard", "scene2/site5")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene2/site3") + \
                        ExpDataset(root_dir, "nohazard", "scene2/site4") + \
                        ExpDataset(root_dir, "nohazard", "scene2/site5")
    elif scene_num == 3:
        hazard_test = ExpDataset(root_dir, "hazard", "scene3/site9") + \
                        ExpDataset(root_dir, "hazard", "scene3/site10") + \
                        ExpDataset(root_dir, "hazard", "scene3/site11") + \
                        ExpDataset(root_dir, "hazard", "scene3/site12") + \
                        ExpDataset(root_dir, "hazard", "scene3/site13") + \
                        ExpDataset(root_dir, "hazard", "scene3/site21") 

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene3/site9") + \
                        ExpDataset(root_dir, "nohazard", "scene3/site10") + \
                        ExpDataset(root_dir, "nohazard", "scene3/site11") + \
                        ExpDataset(root_dir, "nohazard", "scene3/site12") + \
                        ExpDataset(root_dir, "nohazard", "scene3/site13") + \
                        ExpDataset(root_dir, "nohazard", "scene3/site21")
    elif scene_num == 4:
        hazard_test = ExpDataset(root_dir, "hazard", "scene4/site14") + \
                        ExpDataset(root_dir, "hazard", "scene4/site15") + \
                        ExpDataset(root_dir, "hazard", "scene4/site16")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene4/site14") + \
                        ExpDataset(root_dir, "nohazard", "scene4/site15") + \
                        ExpDataset(root_dir, "nohazard", "scene4/site16")
    elif scene_num == 5:
        hazard_test = ExpDataset(root_dir, "hazard", "scene5/site17") + \
            ExpDataset(root_dir, "hazard", "scene5/site18")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene5/site17") + \
            ExpDataset(root_dir, "nohazard", "scene5/site18")
    elif scene_num == 6:
        hazard_test = ExpDataset(root_dir, "hazard", "scene6/site19") + \
                        ExpDataset(root_dir, "hazard", "scene6/site20")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene6/site19") + \
                        ExpDataset(root_dir, "nohazard", "scene6/site20")
    else:
        hazard_test = ExpDataset(root_dir, "hazard", "scene7/site22") + \
                    ExpDataset(root_dir, "hazard", "scene7/site23") + \
                    ExpDataset(root_dir, "hazard", "scene7/site24")

        nohazard_test = ExpDataset(root_dir, "nohazard", "scene7/site22") + \
                    ExpDataset(root_dir, "nohazard", "scene7/site23") + \
                    ExpDataset(root_dir, "nohazard", "scene7/site24")

    test_set = hazard_test + nohazard_test

    test_set_size = test_set.__len__()
    print("The length of test set is {}".format(test_set_size))

    test_dataloader = DataLoader(test_set, batch_size=32, num_workers=8, shuffle=True, drop_last=False)

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

    model_state_dict = torch.load("trainedModels/vgg16_1_source_scene1.pth")
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    model.eval()
    labels = []
    preds = []
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)

            outputs_idx = outputs.argmax(1)
            idxN = len(targets)
            for idx in range(idxN):
                labels.append(targets[idx].item())
                preds.append(outputs_idx[idx].item())

    cm = confusion_matrix(labels, preds)

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

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

    print("Test weighted F1-score is {}".format(wF1Score))
    print("Test NoHazard Accuracy is {}".format(noh_acc))
    print("Test Hazard Accuracy is {}".format(h_acc))