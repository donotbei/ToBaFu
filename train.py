import os
from datetime import datetime

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.utils import read_data
from model.topo_model import get_TOPO_model as TopoModel
from model.img_model import get_Image_model as ImgModel
from tools.tools import EarlyStopping, visualize_losses


def train_model(
        model,
        train_loader,
        val_loader,
        datatype,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        early_stopping=EarlyStopping(),
        num_epochs=100,
        weight_decay=0.01,
        lr=0.001,
        lr_decay=0.1,
        lr_patience=20,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    print(f'Training in {device}')

    # move the model to the device
    model.to(device)

    # define the loss function
    criterion = loss

    # define the optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer(trainable_params, lr=lr, weight_decay=weight_decay)

    # define the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay,
                                  patience=lr_patience, threshold=1e-5, min_lr=1e-8)

    train_losses, val_losses = [], []
    val_accuracies = []
    learning_rates = []
    gradient_norms = []

    # train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_norm = 0.0
        for i, (inputs, labels) in enumerate(train_loader):  # 64*90, 64*1
        # for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 计算梯度范数
            batch_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    batch_norm += param_norm.item() ** 2
            total_norm += batch_norm ** 0.5

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 记录平均梯度范数
        average_norm = total_norm / len(train_loader)
        gradient_norms.append(average_norm)

        # validate the model every epoch
        model.eval()
        val_loss = 0.0
        count = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_loader):
            if datatype == 'topo':
                inputs = inputs[0]
            elif datatype == 'image':
                inputs = inputs[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predict = outputs.argmax(dim=1, keepdim=True)
            count += predict.eq(labels.view_as(predict)).sum().item()
            total += labels.size(0)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        accuracy = count / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        # update the learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # print the validation loss and accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {accuracy * 100:.2f}%')

        # check the early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    return train_losses, val_losses, val_accuracies, learning_rates, gradient_norms


def train_topo_model(
        input_size,
        num_classes,
        config,
        train_loader,
        val_loader,
        ckpt_dir,
):
    weight_decay = config['weight_decay']
    lr = config['learning_rate']
    num_epochs = config['num_epochs']
    num_blocks = config['num_blocks']
    hidden_size = config['hidden_size']
    patience = config['patience']
    lr_patience = config['lr_patience']
    lr_decay = config['lr_decay']
    dropout_rate = config['dropout_rate']

    # define the model
    model = TopoModel(input_size, num_classes, hidden_size, num_blocks, dropout_rate)

    # define the early stopping
    ckpt_path = os.path.join(ckpt_dir, 'topo_checkpoint.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=ckpt_path)

    # train the model
    losses = train_model(model, train_loader, val_loader, datatype='topo',
                         early_stopping=early_stopping, num_epochs=num_epochs, weight_decay=weight_decay,
                         lr=lr, lr_decay=lr_decay, lr_patience=lr_patience)

    # visualize the losses
    visualize_losses(*losses, os.path.join(ckpt_dir, 'topo_losses.png'))

    return


def train_img_model(
        num_classes,
        config,
        train_loader,
        val_loader,
        ckpt_dir,
):
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    lr = config['learning_rate']
    patience = config['patience']
    lr_patience = config['lr_patience']
    lr_decay = config['lr_decay']
    dropout_rate = config['dropout_rate']

    # define the model
    model = ImgModel(num_classes, dropout_rate=dropout_rate)

    # define the early stopping
    ckpt_path = os.path.join(ckpt_dir, 'image_checkpoint.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=ckpt_path)

    # train the model
    losses = train_model(model, train_loader, val_loader, datatype='image', early_stopping=early_stopping,
                         num_epochs=num_epochs, weight_decay=weight_decay, lr=lr, lr_decay=lr_decay, lr_patience=lr_patience)

    # visualize the losses
    visualize_losses(*losses, os.path.join(ckpt_dir, 'image_losses.png'))

    return


def train():
    topo_train_loader, image_train_loader = read_data('train')
    combine_loader = read_data('test')

    config = OmegaConf.load('config/config.yaml')['train']
    topo_config = config['topo']
    image_config = config['image']

    ckpt_dir = config['ckpt_dir']
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topo_feature = config['topo_feature']
    ckpt_dir = os.path.join(ckpt_dir, topo_feature+'_'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)

    # get the input size and num_classes
    features, labels = next(iter(topo_train_loader))
    input_size = features.shape[1]
    num_classes = len(set(labels.numpy()))

    train_topo_model(input_size, num_classes, topo_config, topo_train_loader, combine_loader, ckpt_dir)
    train_img_model(num_classes, image_config, image_train_loader, combine_loader, ckpt_dir)


if __name__ == '__main__':
    train()
