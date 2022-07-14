import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import timeit
import seaborn as sns
import pickle
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import glob
import os
from tqdm import tqdm
import utils


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

def load_dataset(root_dir="data"):
    data = []
    label = []

    for filename in glob.glob(root_dir + "/person*.png"):
        img=Image.open(filename)

        if img.mode == "L":  # for grayscale images
            img = np.array(img)
            img = np.moveaxis(np.stack((img,img,img)), 0, 2)
            img = Image.fromarray(img)

        data.append(img)
        label.append(0)

    for filename in glob.glob(root_dir + "/robot*.png"):
        img=Image.open(filename)

        if img.mode == "L":  # for grayscale images
            img = np.array(img)
            img = np.moveaxis(np.stack((img,img,img)), 0, 2)
            img = Image.fromarray(img)

        data.append(img)
        label.append(1)

    return np.array(data), np.array(label)


@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):

    total_loss = 0
    total_accuracy = 0
    confusion_matrix = np.zeros((10,10))
    
    for (val_images, val_labels) in dataloader:

        val_images = val_images.to(device)
        val_labels = val_labels.flatten().to(device)

        #Loss
        prediction = model(val_images)
        loss = loss_function(prediction, val_labels).item()
        total_loss += loss

        #Accuracy
        pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        accuracy = np.mean(pred_label == val_labels.detach().cpu().numpy())
        total_accuracy += accuracy
        
        #Confusion matrix
        for y, y_pred in zip(val_labels.detach().cpu().numpy(), pred_label):
            confusion_matrix[int(y_pred)][int(y)] += 1

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    return avg_loss, avg_accuracy, confusion_matrix

def train_one_epoch(model, dataloader, optimizer, loss_function, device=torch.device('cpu'), debug=False):
    weight_grad_norm_hist = []
    bias_grad_norm_hist = []
    total_loss = 0
    total_accuracy = 0
    
    #progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, (trainingImages,trainingLabels) in enumerate(dataloader):
        optimizer.zero_grad()

        trainingImages = trainingImages.to(device)
        trainingLabels = trainingLabels.flatten().to(device)

        prediction = model(trainingImages)
        
        loss = loss_function(prediction, trainingLabels)
        loss_without_reg = loss.detach().cpu().numpy()
        total_loss += loss_without_reg
        
        loss.backward()
        optimizer.step()

        if debug:
            with torch.no_grad():
                if index % 100 == 0:
                    pass
                    #print('Train {}/{} Loss {:.6f}'.format(index, len(dataloader), loss_without_reg))
        
        #Accuracy
        #pred_label = np.argmax(prediction.detach().cpu().numpy(), axis=1)
        #accuracy = np.mean(pred_label == trainingLabels.detach().cpu().numpy())
        #total_accuracy += accuracy
        
        #Accuracy
        pred_label = torch.argmax(prediction, dim=1)
        accuracy = ((pred_label == trainingLabels).sum().item()) / len(trainingImages)
        total_accuracy += accuracy
        
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
        
    return avg_loss, avg_accuracy  #, weight_grad_norm_hist, bias_grad_norm_hist

def train(model, train_loader, val_loader, optimizer, loss_function, max_epoch=30, device=torch.device('cpu'), debug=False):

    training_loss_hist = []
    training_acc_hist = []
    valid_loss_hist = []
    valid_acc_hist = []
    start_time = timeit.default_timer()

    progress_bar = tqdm(range(max_epoch))

    for epoch in progress_bar:

        #training
        train_loss, train_acc = train_one_epoch(model, 
                                                    train_loader, 
                                                    optimizer, 
                                                    loss_function,
                                                    device=device, 
                                                    debug=debug)
        #evaluation
        valid_loss, valid_acc, confusion_matrix = evaluate(model, val_loader, loss_function, device=device)

        progress_bar.set_description("epoch {} train loss:{} train acc:{} val loss:{} val acc:{}".format(epoch, round(train_loss,5), round(train_acc,5), round(valid_loss,5), round(valid_acc,5)))
        #f"Iter {itr}: loss {loss.detach().cpu().numpy():.5f}. "

        #loss and accuracy histories are being kept
        training_loss_hist.append(train_loss)
        training_acc_hist.append(train_acc)
        valid_loss_hist.append(valid_loss)  
        valid_acc_hist.append(valid_acc)

    print("Time: ", timeit.default_timer()-start_time)
    
    return training_loss_hist, training_acc_hist, valid_loss_hist, valid_acc_hist

##################### Visualization Functions ############################
def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f

def visualize_loss_acc(training_loss_hist, valid_loss_hist, training_acc_hist, valid_acc_hist):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 5))
    
    # todo: _=plt.plot(val_epochs, smooth(val_acc, 31), label="Smoothed")

    ax1.plot(training_loss_hist, label="Traning Loss")
    ax1.plot(valid_loss_hist, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(training_acc_hist, label="Traning Accuracy")
    ax2.plot(valid_acc_hist, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.tight_layout()

############################################################################

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
def create_params_to_update(model, feature_extract, debug=False):
    params_to_update = model.parameters()
    if debug:
        print("Params to learn:")
        
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if debug:
                    print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                if debug:
                    print("\t",name)
    
    return params_to_update

############################################################################

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)

        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False

        in_features = model_ft.classifier[0].in_features
        
        """
        model_ft.classifier = nn.Sequential(
                  nn.Linear(in_features=in_features, out_features=512, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5, inplace=False),
                  nn.Linear(in_features=512, out_features=128, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Dropout(p=0.5, inplace=False),
                  nn.Linear(in_features=128, out_features=num_classes, bias=True),
                )
        """
        model_ft.classifier = nn.Linear(in_features, num_classes) # replace the big MLP with a single layer
        
    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
                
        in_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False
                
        in_features = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(in_features, num_classes)

    else:
        raise Exception("Invalid model name!")

    return model_ft