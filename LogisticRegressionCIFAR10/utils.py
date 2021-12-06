import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

def train_one_epoch(model, dataloader, optimizer, loss_function, device=torch.device('cpu'), debug=False):
    weight_grad_norm_hist = []
    bias_grad_norm_hist = []
    total_loss = 0
    total_accuracy = 0
    
    for index, (trainingImages,trainingLabels) in enumerate(dataloader):
        optimizer.zero_grad()

        trainingImages = trainingImages.to(device)
        trainingLabels = trainingLabels.to(device)

        prediction = model(trainingImages.reshape(-1, 32*32*3))
        
        loss = loss_function(prediction, trainingLabels)
        total_loss += loss.detach().cpu().numpy()
        
        loss.backward()
        optimizer.step()

        if debug:
            with torch.no_grad():
                if index % 100 == 0:
                    print('Train {}/{} Loss {:.6f}'.format(index, len(dataloader), loss.item()))
        
        weight_grad_norm_hist.append(np.linalg.norm(model.linear1.weight.grad.detach().cpu().numpy()))
        bias_grad_norm_hist.append(np.linalg.norm(model.linear1.bias.grad.detach().cpu().numpy()))
                    
        #Accuracy
        pred_label = torch.argmax(prediction, dim=1)
        accuracy = ((pred_label == trainingLabels).sum().item()) / len(trainingImages)
        total_accuracy += accuracy
        
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
        
    return avg_loss, avg_accuracy, weight_grad_norm_hist, bias_grad_norm_hist

def evaluate(model, dataloader, loss_function, device):

    total_loss = 0
    total_accuracy = 0
    confusion_matrix = np.zeros((10,10))
    
    for (val_images, val_labels) in dataloader:

        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        #Loss
        prediction = model(val_images.reshape(-1, 32*32*3))
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

def train_model(model, train_loader, val_loader, optimizer, loss_function, max_epoch, device):
    training_loss_hist = []
    training_acc_hist = []
    valid_loss_hist = []
    valid_acc_hist = []
    weight_grad_norm_hist = []
    bias_grad_norm_hist = []

    start_time = timeit.default_timer()

    for epoch in range(max_epoch):
        
        #training
        train_loss, train_acc, weight_grad_norm_hist, bias_grad_norm_hist = train_one_epoch(model, 
                                                                                            train_loader, 
                                                                                            optimizer, 
                                                                                            loss_function, 
                                                                                            device=device, 
                                                                                            debug=False)
        #evaluation
        valid_loss, valid_acc, confusion_matrix = evaluate(model, val_loader, loss_function, device=device)
        
        print("epoch {} train loss: {} train acc: {} val loss: {} val acc: {}".format(epoch, round(train_loss,5), round(train_acc,5), round(valid_loss,5), round(valid_acc,5)))

        #loss and accuracy histories are being kept
        training_loss_hist.append(train_loss)
        training_acc_hist.append(train_acc)
        valid_loss_hist.append(valid_loss)  
        valid_acc_hist.append(valid_acc)
        weight_grad_norm_hist.extend(weight_grad_norm_hist)
        bias_grad_norm_hist.extend(bias_grad_norm_hist)
        
    end_time = timeit.default_timer()

    print("Time: ", end_time-start_time)

    return training_loss_hist, training_acc_hist, valid_loss_hist, valid_acc_hist, weight_grad_norm_hist, bias_grad_norm_hist

def plot_learning_curve(training_loss_hist, valid_loss_hist, training_acc_hist, valid_acc_hist):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16, 5))

    ax1.plot(training_loss_hist, label="Traning Loss")
    ax1.plot(valid_loss_hist, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(training_acc_hist, label="Traning Accuracy")
    ax2.plot(valid_acc_hist, label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax2.legend()
    fig.tight_layout()

def plot_gradient_norms(weight_grad_norm_hist, bias_grad_norm_hist, max_epoch):
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 7))

    epochs = np.linspace(0, max_epoch, len(weight_grad_norm_hist))
    batches_per_epoch = len(epochs) / max_epoch

    ax1.plot(epochs, weight_grad_norm_hist, linewidth=0.3)
    ax1.set_title("L2-Norm of the Weight's Gradients (without Bias)")
    ax2.plot(epochs, bias_grad_norm_hist, linewidth=0.3)
    ax2.set_title("L2-Norm of the Weight's Gradients (only Bias)")
    fig.tight_layout()

def plot_confusion_matrix(confusion_matrix, classes):
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt="g", ax=ax, xticklabels=classes, yticklabels=classes)
    plt.xlabel("True Class")
    plt.ylabel("prediction")
    _=plt.title("Confusion Matrix")
