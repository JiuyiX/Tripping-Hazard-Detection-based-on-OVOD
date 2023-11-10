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

    parser.add_argument('--TRAIN_DIR',default='../TH_in_outdoor_source/indoor',help='The training image directory')
    parser.add_argument('--TEST_DIR',default='../TH_in_outdoor_source/outdoor',help='The test image directory')
    parser.add_argument('--IS_AUG',default='1',help='Decide whether to use data augmentation or not')
    parser.add_argument('--MODEL',default='VGG16',help='The classification model')
    
    args = parser.parse_args()

    train_dir = args.TRAIN_DIR
    test_dir = args.TEST_DIR
    is_aug = int(args.IS_AUG)
    model_name = args.MODEL

    epoch = 150

    if is_aug:
        print("Data augmentation starts...")
        hazard_train = ExpDataset_aug(train_dir, "hazard", "scene1") + \
            ExpDataset_aug(train_dir, "hazard", "scene2") + \
            ExpDataset_aug(train_dir, "hazard", "scene3") + \
            ExpDataset_aug(train_dir, "hazard", "scene4") + \
            ExpDataset_aug(train_dir, "hazard", "scene5") + \
            ExpDataset_aug(train_dir, "hazard", "scene6") + \
            ExpDataset_aug(train_dir, "hazard", "scene7")
        nohazard_train = ExpDataset_aug(train_dir, "nohazard", "scene1") + \
            ExpDataset_aug(train_dir, "nohazard", "scene2") + \
            ExpDataset_aug(train_dir, "nohazard", "scene3") + \
            ExpDataset_aug(train_dir, "nohazard", "scene4") + \
            ExpDataset_aug(train_dir, "nohazard", "scene5") + \
            ExpDataset_aug(train_dir, "nohazard", "scene6") + \
            ExpDataset_aug(train_dir, "nohazard", "scene7")
    else:
        hazard_train = ExpDataset(train_dir, "hazard", "scene1") + \
            ExpDataset(train_dir, "hazard", "scene2") + \
            ExpDataset(train_dir, "hazard", "scene3") + \
            ExpDataset(train_dir, "hazard", "scene4") + \
            ExpDataset(train_dir, "hazard", "scene5") + \
            ExpDataset(train_dir, "hazard", "scene6") + \
            ExpDataset(train_dir, "hazard", "scene7")
        nohazard_train = ExpDataset(train_dir, "nohazard", "scene1") + \
            ExpDataset(train_dir, "nohazard", "scene2") + \
            ExpDataset(train_dir, "nohazard", "scene3") + \
            ExpDataset(train_dir, "nohazard", "scene4") + \
            ExpDataset(train_dir, "nohazard", "scene5") + \
            ExpDataset(train_dir, "nohazard", "scene6") + \
            ExpDataset(train_dir, "nohazard", "scene7")

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

    train_set = hazard_train + nohazard_train
    test_set = hazard_test + nohazard_test

    train_set_size = train_set.__len__()
    test_set_size = test_set.__len__()
    print("The length of train set is {}".format(train_set_size))
    print("The length of test set is {}".format(test_set_size))

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
            modelname = f"trainedModels/test_in_outdoor.pth"
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
    plt.title('Train Loss for Test In_Outdoor')
    plt.savefig(f"experimentGraphs/train_loss_in_outdoor.jpg")
    plt.close()

    plt.figure()
    plt.plot(xs, epoch_test_loss, 'b', label='loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Test Loss for Test In_Outdoor')
    plt.savefig(f"experimentGraphs/test_loss_in_outdoor.jpg")
    plt.close()

    print("Best confusion matrix is {}".format(best_cm))

    TP = best_cm[0][0]
    FN = best_cm[0][1]
    FP = best_cm[1][0]
    TN = best_cm[1][1]

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