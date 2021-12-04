import yaml
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

from models import SegNet, SegNet_3D, UNet3D, VNet, UNet3D_LSTM
from dataloader import get_train_loaders
from utils import get_logger


def get_model(model_name):
    m = importlib.import_module('models')
    model = getattr(m, model_name)
    return model


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def train(config, loaders):
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    wd = config['training']['wd']
    model_name = config['training']['model_name']
    in_channels = config['training']['in_channels']
    out_channels = config['training']['out_channels']
    layer_order = config['training']['layer_order']
    f_maps = config['training']['f_maps']
    num_groups = config['training']['num_groups']
    pool_kernel_size = tuple(config['training']['pool_kernel_size'])
    final_sigmoid = config['training']['final_sigmoid']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = get_logger('Training')
    logger.info(f'Using {device} for training')

    # weights for CrossEntropyLoss
    for images, labels in loaders['train']:
        labels[labels > 0] = 1.
        l = labels[0, 0]
        break
    w1 = l[l != 0].shape[0] / (l.shape[0] * l.shape[1])
    w0 = l[l == 0].shape[0] / (l.shape[0] * l.shape[1])
    class_freq = (w0, w1)
    weights = torch.FloatTensor(np.median(class_freq) / class_freq).to(device)
    logger.info(f'Weights {class_freq} for CrossEntropyLoss')
    ce_criterion = nn.CrossEntropyLoss(weight=weights)
    mse_criterion = nn.MSELoss(reduction='sum')

    logger.info(f'Model: {model_name}')
    m = get_model(model_name)
    model = m(in_channels=in_channels, out_channels=out_channels,
              layer_order=layer_order, f_maps=f_maps,
              num_groups=num_groups, pool_kernel_size=pool_kernel_size,
              final_sigmoid=final_sigmoid, device=device).to(device)
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    train_loss, val_loss = [], []
    train_ce, val_ce = [], []
    train_mse, val_mse = [], []
    plt_train_loss, plt_val_loss = [], []

    logger.info(f'Number of epochs {num_epochs}')
    for epoch in range(num_epochs):

        start_time = time.time()
        model.train()
        for images, labels in loaders['train']:
            images = images[None, :].to(device, torch.float32)
            labels[labels > 0] = 1.
            labels = labels.long().to(device)

        optimizer.zero_grad()
        outputs, encoder_outputs, lstm_outputs = model(images)
        loss_ce = ce_criterion(outputs, labels)
        loss_mse = mse_criterion(outputs[:, 1], labels.float())
        loss = loss_mse + loss_ce
        loss.backward()
        optimizer.step()

        train_ce.append(loss_ce.item())
        train_mse.append(loss_mse.item())
        train_loss.append(loss.item())

        model.train(False)
        for images, labels in loaders['val']:
            images = images[None, :].to(device, torch.float32)
            labels[labels > 0] = 1.
            labels = labels.long().to(device)
            outputs, encoder_outputs, lstm_outputs = model(images)
            loss_ce = ce_criterion(outputs, labels)
            loss_mse = mse_criterion(outputs[:, 1], labels.float())
            loss = loss_mse + loss_ce

            val_mse.append(loss_mse.item())
            val_ce.append(loss_ce.item())
            val_loss.append(loss.item())

        logger.info("Epoch {}/{} took {:.2f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        logger.info("Loss Train: {:.3f}, Val: {:.3f}".format(np.mean(train_loss[-len(loaders['train']) // batch_size:]),
                    np.mean(val_loss[-len(loaders['val']) // batch_size:])))
        logger.info("CE Train: {:.3f}, Val: {:.3f}".format(np.mean(train_ce[-len(loaders['train']) // batch_size:]),
                    np.mean(val_ce[-len(loaders['val']) // batch_size:])))
        logger.info("MSE Train : {:.3f}, Val: {:.3f}".format(np.mean(train_mse[-len(loaders['train']) // batch_size:]),
                    np.mean(val_mse[-len(loaders['val']) // batch_size:])))

    logger.info(f'Reached maximum number of epochs: {num_epochs}. Finishing training.')

    dir_models = './models/'
    if not os.path.exists(dir_models):
        os.makedirs(dir_models)
    model_save_name = config['training']['model_save_name']
    path = F"./{dir_models + model_save_name}"
    torch.save(model.state_dict(), path)
    logger.info(f'Save model in {path}')


def main():
    # Load experiment configuration
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Load data
    loaders = get_train_loaders(config)
    # Start training
    train(config, loaders)


if __name__ == '__main__':
    main()