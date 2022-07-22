import torch; import torch.nn as nn
import optuna
import numpy as np
import matplotlib; import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image
import os
import random
import pickle
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from tqdm import tqdm


#########################################################
##### HEADER ############################################
#########################################################


def initialize_notebook():
    print("Library Versions:")
    print("- PyTorch:", torch.__version__)
    print("- Optuna:", optuna.__version__)
    print("- Numpy:", np.__version__)
    print("- Matplotlib:", matplotlib.__version__)
    print("- Seaborn:", sns.__version__)
    print("- PIL:", PIL.__version__)
    print("")
    print("More Information:")
    print("- torch.cuda.is_available() ->", torch.cuda.is_available())
    plt.style.use('seaborn')
    print("- plt.style.use() <- 'seaborn'")



#########################################################
##### TRAIN-EVAL ########################################
#########################################################

def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """
    Training a model for one epoch

    Returns:
        mean_loss, loss_list
    """
    
    loss_list = []
    #progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)
         
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        #progress_bar.set_description(f"Epoch {epoch+1} Iter {i+1}: loss {loss.item():.5f}. ")
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list


@torch.no_grad()
def eval_model(model, eval_loader, criterion, device):
    """
    Evaluating the model for either validation or test

    Returns:
        accuracy, loss
    """

    correct = 0
    total = 0
    loss_list = []
    
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass only to get logits/output
        outputs = model(images)
                 
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
            
        # Get predictions from the maximum value
        preds = torch.argmax(outputs, dim=1)
        correct += len( torch.where(preds==labels)[0] )
        total += len(labels)
                 
    # Total correct predictions and loss
    accuracy = correct / total * 100
    loss = np.mean(loss_list)
    
    return accuracy, loss


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader, num_epochs, device):
    """
    Training a model for a given number of epochs

    Args:
        scheduler: For example; torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                   Should be connected to the optimizer, also can be None.

    Returns:
        train_loss, val_loss, loss_iters, valid_acc
    """
    
    train_loss = []
    val_loss =  []
    loss_iters = []
    valid_acc = []
    
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
           
        # validation epoch
        model.eval()  # important for dropout and batch norms
        accuracy, loss = eval_model(
                    model=model, eval_loader=valid_loader,
                    criterion=criterion, device=device)
        valid_acc.append(accuracy)
        val_loss.append(loss)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        if scheduler is not None:
            scheduler.step()
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters
        
        progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} Train loss: {round(mean_loss, 5)} Valid loss: {round(loss, 5)} Accuracy: {accuracy}%")

    return train_loss, val_loss, loss_iters, valid_acc


#########################################################
##### PLOT ##############################################
#########################################################


def smooth(f, K=5):
    """
    Smoothing a function using a low-pass filter (mean) of size K

    Returns:
        smooth_f
    """

    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def plot_loss_train_val(loss_iters, train_loss, val_loss, valid_acc):
    """
    Plots 3 graphs; loss/iteraions, train_loss/epoch, val_loss/epoch respectively and indicates validation accuracy
    """

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")

    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs[1:], train_loss[1:], c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs[1:], val_loss[1:], c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")

    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs[1:], valid_acc[1:], c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Validation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})") # todo: max?

    plt.show()


def show_grid(dataset, label_map=None):
    """
    Imshow for Tensor.

    Args:
        dataset: Give dataset not dataloader.
        label_map: Array type mapping. Maps label values to int/str for visualization
    """
    
    plt.figure(figsize=(8*2, 4*2))
    max_i = 32
    for i, sample in enumerate(dataset):
        if i >= max_i:
            break
        image, label = sample
        plt.subplot(4,8,i+1)

        if isinstance(image, PIL.Image.Image):
            plt.imshow(image)
        elif isinstance(image, torch.Tensor):
            plt.imshow(image.numpy().transpose((1, 2, 0)))
        else:
            print("Unknown datatype:", type(image))
            break

        plt.axis("off")
        if label_map is not None:
            plt.title("Label: " + str(label_map[label]))
        else:
            plt.title("Label: " + str(label))
    plt.tight_layout()

@torch.no_grad()
def calculate_confusion_matrix(model, data_loader, num_classes=10):
    confusion_matrix = np.zeros((num_classes,num_classes))
    model_device = next(model.parameters()).device
    model.eval()
    for batch, label in data_loader:
        output = model(batch.to(model_device))
        
        prediction = torch.argmax(output, dim=1).detach().cpu().numpy()
        true_value = label.numpy()
        
        for i,j in zip(prediction, true_value):
            confusion_matrix[i,j] +=1
    
    return confusion_matrix

def visualize_confusion_matrix(confusion_matrix, labels=None, ax=None):
    """
    Plots confusion matrix

    Args:
        confusion_matrix: 2D numpy array
        labels: Labels for confusion matrix items
        ax: Matplotlib axis

    Returns:
        fig, ax (fig is None if ax is not given)
    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    if labels is None:
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", ax=ax)
    else:
        sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", ax=ax, xticklabels=labels, yticklabels=labels)

    ax.set_title("Confusion Matrix")
    return fig, ax

def confusion_matrix_3(model, train_loader, val_loader, test_loader, data_labels):
    fig, axs = plt.subplots(1,3, figsize=(15,4))
    
    confusion_matrix = calculate_confusion_matrix(model, train_loader)
    _=visualize_confusion_matrix(confusion_matrix, data_labels, ax=axs[0])
    _=axs[0].set_title("Confusion Matrix on Train Data")

    confusion_matrix = calculate_confusion_matrix(model, val_loader)
    _=visualize_confusion_matrix(confusion_matrix, data_labels, ax=axs[1])
    _=axs[1].set_title("Confusion Matrix on Validation Data")

    confusion_matrix = calculate_confusion_matrix(model, test_loader)
    _=visualize_confusion_matrix(confusion_matrix, data_labels, ax=axs[2])
    _=axs[2].set_title("Confusion Matrix on Test Data")

    fig.tight_layout()

#########################################################
##### UTILS #############################################
#########################################################

def count_model_params(model):
    """
    Counting the number of learnable parameters in a nn.Module

    Returns:
        num_params
    """

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def save_model(model, optimizer, epoch, stats):
    """ Saving model checkpoint """
    
    if(not os.path.exists("models")):
        os.makedirs("models")
    savepath = f"models/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """

    """
    Loading pretrained checkpoint

    Returns:
        model, optimizer, epoch, stats
    """
    
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats


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


def get_device():
    """
    Returns CUDA device if possible, otherwise returns cpu

    Returns:
        device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device