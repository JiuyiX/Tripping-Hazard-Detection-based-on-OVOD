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

    parser.add_argument('--TEST_DIR',default='../TH_in_outdoor_source/outdoor',help='The test image directory')
    parser.add_argument('--MODEL',default='VGG16',help='The classification model')

    args = parser.parse_args()

    test_dir = args.TEST_DIR
    model_name = args.MODEL

    hazard_test = ExpDataset(test_dir, "hazard", "scene1") + \
        ExpDataset(test_dir, "hazard", "scene2") + \
        ExpDataset(test_dir, "hazard", "scene3") + \
        ExpDataset(test_dir, "hazard", "scene4") + \
        ExpDataset(test_dir, "hazard", "scene5") + \
        ExpDataset(test_dir, "hazard", "scene6") + \
        ExpDataset(test_dir, "hazard", "scene7")
    nohazard_test = ExpDataset(test_dir, "nohazard", "scene1") + \
        ExpDataset(test_dir, "nohazard", "scene2") + \
        ExpDataset(test_dir, "nohazard", "scene3") + \
        ExpDataset(test_dir, "nohazard", "scene4") + \
        ExpDataset(test_dir, "nohazard", "scene5") + \
        ExpDataset(test_dir, "nohazard", "scene6") + \
        ExpDataset(test_dir, "nohazard", "scene7")
    
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

    model_state_dict = torch.load("trainedModels/vgg16_1_indoor_source.pth")
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    testSet_1 = ExpDataset(test_dir, "hazard", "scene1") + ExpDataset(test_dir, "nohazard", "scene1")
    testSet_2 = ExpDataset(test_dir, "hazard", "scene2") + ExpDataset(test_dir, "nohazard", "scene2")
    testSet_3 = ExpDataset(test_dir, "hazard", "scene3") + ExpDataset(test_dir, "nohazard", "scene3")
    testSet_4 = ExpDataset(test_dir, "hazard", "scene4") + ExpDataset(test_dir, "nohazard", "scene4")
    testSet_5 = ExpDataset(test_dir, "hazard", "scene5") + ExpDataset(test_dir, "nohazard", "scene5")
    testSet_6 = ExpDataset(test_dir, "hazard", "scene6") + ExpDataset(test_dir, "nohazard", "scene6")
    testSet_7 = ExpDataset(test_dir, "hazard", "scene7") + ExpDataset(test_dir, "nohazard", "scene7")
    
    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    TN_sum = 0

    model.eval()
    if testSet_1.__len__() > 30:
        testSet1_dataloader = DataLoader(testSet_1, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        with torch.no_grad():
            for data in testSet1_dataloader:
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
        print("Test of Scene 1:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))
    
    if testSet_2.__len__() > 30:
        testSet2_dataloader = DataLoader(testSet_2, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        with torch.no_grad():
            for data in testSet2_dataloader:
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
        print("Test of Scene 2:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    if testSet_3.__len__() > 30:
        testSet3_dataloader = DataLoader(testSet_3, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        with torch.no_grad():
            for data in testSet3_dataloader:
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
        print("Test of Scene 3:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    if testSet_4.__len__() > 30:
        testSet4_dataloader = DataLoader(testSet_4, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        with torch.no_grad():
            for data in testSet4_dataloader:
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
        print("Test of Scene 4:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    if testSet_5.__len__() > 30:
        testSet5_dataloader = DataLoader(testSet_5, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        for data in testSet5_dataloader:
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
        print("Test of Scene 5:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    if testSet_6.__len__() > 30:
        testSet6_dataloader = DataLoader(testSet_6, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        for data in testSet6_dataloader:
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
        print("Test of Scene 6:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    if testSet_7.__len__() > 30:
        testSet7_dataloader = DataLoader(testSet_7, batch_size=32, num_workers=8, shuffle=True, drop_last=False)
        labels = []
        preds = []
        for data in testSet7_dataloader:
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
        print("Test of Scene 7:")

        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

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

        print("Test weighted F1-score is {}".format(wF1Score))
        print("Test NoHazard Accuracy is {}".format(noh_acc))
        print("Test Hazard Accuracy is {}".format(h_acc))

    print("Evaluation Report:")
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
    print("Best weighted F1-score on the test set is {}".format(wF1Score_all))
    print("Best NoHazard Accuracy on the test set is {}".format(noh_acc_all))
    print("Best Hazard Accuracy on the test set is {}".format(h_acc_all))