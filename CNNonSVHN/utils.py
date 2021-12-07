import numpy as np
import torch
import os
import random
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

@torch.no_grad()
def evaluate(model, dataloader, loss_function, device):

    total_loss = 0
    total_accuracy = 0
    confusion_matrix = np.zeros((10,10))
    
    for (val_images, val_labels) in dataloader:

        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

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

def train_one_epoch(model, dataloader, optimizer, loss_function, reg_function=None, reg_params=None, device=torch.device('cpu'), debug=False):
    weight_grad_norm_hist = []
    bias_grad_norm_hist = []
    total_loss = 0
    total_accuracy = 0
    
    for index, (trainingImages,trainingLabels) in enumerate(dataloader):
        optimizer.zero_grad()

        trainingImages = trainingImages.to(device)
        trainingLabels = trainingLabels.to(device)

        prediction = model(trainingImages)
        
        loss = loss_function(prediction, trainingLabels)
        loss_without_reg = loss.detach().cpu().numpy()
        total_loss += loss_without_reg
        
        if reg_function is not None:
            loss = torch.add(loss, reg_function(model, reg_params))
        
        loss.backward()
        optimizer.step()

        if debug:
            with torch.no_grad():
                if index % 100 == 0:
                    print('Train {}/{} Loss {:.6f}'.format(index, len(dataloader), loss_without_reg))
                    
        #Accuracy
        pred_label = torch.argmax(prediction, dim=1)
        accuracy = ((pred_label == trainingLabels).sum().item()) / len(trainingImages)
        total_accuracy += accuracy
        
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
        
    return avg_loss, avg_accuracy  #, weight_grad_norm_hist, bias_grad_norm_hist

def train(model, train_loader, val_loader, optimizer, loss_function, reg_function=None, reg_params=None, max_epoch=30, device=torch.device('cpu'), debug=False):

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
                                                    reg_function=reg_function,
                                                    reg_params=reg_params,
                                                    device=device, 
                                                    debug=debug)
        #evaluation
        valid_loss, valid_acc, confusion_matrix = evaluate(model, val_loader, loss_function, device=device)

        #print("epoch {} train loss: {} train acc: {} val loss: {} val acc: {}".format(epoch, round(train_loss,5), round(train_acc,5), round(valid_loss,5), round(valid_acc,5)))
        progress_bar.set_description("epoch {} train acc: {} val acc: {} train loss: {} val loss: {}".format(epoch, round(train_acc,4), round(valid_acc,4), round(train_loss,4), round(valid_loss,4)))

        #loss and accuracy histories are being kept
        training_loss_hist.append(train_loss)
        training_acc_hist.append(train_acc)
        valid_loss_hist.append(valid_loss)  
        valid_acc_hist.append(valid_acc)

    print("Time: ", timeit.default_timer()-start_time)
    
    return training_loss_hist, training_acc_hist, valid_loss_hist, valid_acc_hist

###################### Visualization Functions ###########################

def visualize_data(trainset):
    # visualize
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
    max_rows, max_cols = 6, 6
    fig, axs = plt.subplots(max_rows, max_cols, figsize=(16,16))
    axs = np.array(axs).flatten()
    for idx, (img,label) in enumerate(train_loader):
        #print("img.shape: {} label: {}".format(img.shape, label))
        axs[idx].imshow(img[0].moveaxis(0,2))
        axs[idx].set_axis_off()
        axs[idx].set_title("label: {}".format(label.item()))
        if idx == max_rows*max_cols-1:
            break
    fig.tight_layout()

def visualize_loss_acc(training_loss_hist, valid_loss_hist, training_acc_hist, valid_acc_hist, title=""):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 5))

    ax1.plot(training_loss_hist, label="Training Loss")
    ax1.plot(valid_loss_hist, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(training_acc_hist, label="Training Accuracy")
    ax2.plot(valid_acc_hist, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

def visualize_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", ax=ax)
    ax.set_title("Confusion Matrix")

def plot_all(scenarios):
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16, 15))
    for s in scenarios:
        title, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = s
        
        print("{} Test Accuracy: {}".format(title, round(test_acc, 3)))
        
        #ax1.plot(train_loss, label=title+" Train Loss")
        ax1.plot(val_loss, label=title+" Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.legend()

        #ax2.plot(train_acc, label=title+" Train Acc")
        ax2.plot(val_acc, label=title+" Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()

        fig.tight_layout()

def visualize_weights(data, title=""):
    cols = data.shape[1]
    rows = data.shape[0]
    
    fig, axs = plt.subplots(rows, cols, figsize=(10,14))
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(data[r, c], cmap="gray")
            axs[r, c].set_title("Nr-"+str(r)+" Ch-"+str(c))
            count+=1

    fig.suptitle(title)
    fig.tight_layout()

def visualize_activations(data, title=""):
    cols = 4
    rows = data.shape[1] // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(10,10))
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(data[0, count], cmap="gray")
            axs[r, c].set_title("Nr-"+str(count))
            count+=1

    fig.suptitle(title)
    fig.tight_layout()
    

###################### Regularization Functions ###########################

def L1_regularization(model, reg_params={"lambda1":0.00001}):
    lambda1 = reg_params["lambda1"]
    
    #weights are obtained since biases are not used for regularization
    weights = []
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            weights.append(param.flatten())
    weights = torch.cat(weights, dim=0)
    
    return lambda1 * torch.norm(weights, p=1)

def L2_regularization(model, reg_params={"lambda2":0.001}):
    lambda2 = reg_params["lambda2"]
    
    #weights are obtained since biases are not used for regularization
    weights = []
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            weights.append(param.flatten())
    weights = torch.cat(weights, dim=0)
    
    return lambda2 * torch.norm(weights, p=2)

def elastic_regularization(model, reg_params={"p":0.5, "lambda1":0.00001, "lambda2":0.001}):
    p = reg_params["p"]
    return p * L1_regularization(model) + (1-p) * L2_regularization(model)
