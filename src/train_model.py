import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.dataloader import dataloader

def prepare_trained_model(args, model, save_path):
    if os.path.isfile(args.model_path):
        print("no need to train.")
        if args.best_model : 
            checkpoint = torch.load(f'{args.model_path}')
            trained_model_state_dict = checkpoint['model_state_dict']
            model.load_state_dict(trained_model_state_dict)
            print('best weights successfully loaded')
        else :
            model.load_state_dict(torch.load(args.model_path))
            print('last weights successfully loaded')
        return model.to(args.device)
    else:
        train(args, model, save_path)
        return model

def train(args, model, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_loader, test_loader, _ = dataloader()

    model = model.to(args.device)

    # defualt from paper
    NUM_EPOCHS = 5
    criterion = nn.CrossEntropyLoss()
    lr = 0.1
    lr_decay = 0.1
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    best_loss = np.inf
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * images.size(0)
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_acc += (outputs.max(1)[1] == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        # save checkpoint
        save_dict = {
            'model_state_dict' : model.state_dict(),
            'epoch' : epoch + 1
        }

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_path = os.path.join(save_path, 'best_trained_model')
            torch.save(save_dict, best_path)

        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    last_path = os.path.join(save_path, 'last_trained_model')
    torch.save(model.state_dict(), last_path)

    if args.best_model : 
        checkpoint = torch.load(best_path)
        trained_model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(trained_model_state_dict)
        print('Best epoch successfully loaded')
        model = model.to(args.device)
    
    return model
