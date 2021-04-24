import argparse
import json
import numpy as np
import os
import sagemaker_containers
import torch
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from model import CNNClassifier

def model_fn(model_dir):
    '''Function to load the PyTorch model from the `model_dir` directory.
            
    Args:
        model_dir (str): directory where the model is located
    
    Returns:
        model (CNNClassifier): CNN model
    
    '''

    print('Loading model.')

    # Determine the device and construct the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNClassifier()

    # Load the stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print('Done loading model.')
    return model

def _get_data_loaders(batch_size, training_dir, mean, std, val_split):
    '''Function to create a data loader by combining the dataset located 
    in the `training_dir` directory with the 'batch_size' sampler to 
    provide an iterable over the given dataset.
            
    Args:
        batch_size (int): how many samples per batch to load
        training_dir (str): directory where the training set is located
        val_split (str): size of the validation set to be used   
    
    Returns:
        DataLoader: an iterable object for the provided training dataset
    
    '''

    print('Split dataset for train and val.')

    transform = T.Compose([T.Resize(112),
                           T.Resize((112, 160)),
                           T.ToTensor(),
                           T.Normalize(mean=mean,
                                       std=std)])

    dataset = datasets.ImageFolder(training_dir, 
                                   transform=transform)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.shuffle(indices)

    idx_train, idx_val = indices[split:], indices[:split]

    sampler_train = SubsetRandomSampler(idx_train)
    sampler_val = SubsetRandomSampler(idx_val)

    print('Get train data loader.')

    loader_train = DataLoader(dataset, 
                              batch_size=batch_size, 
                              sampler=sampler_train)

    print('Get val data loader.')

    loader_val = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=sampler_val)
    
    loaders = {'train': loader_train, 'val': loader_val}

    return loaders


def train(model, loaders, epochs, optimizer, loss_fn, device):
    '''Function called by the PyTorch to kick off training.
            
    Args:
        model (CNNClassifier): the PyTorch model to be trained
        loaders (list): a list of the PyTorch DataLoaders to be used 
            during training and validation
        epochs (int): the total number of epochs to train for
        optimizer (str): the optimizer to use during training
        loss_fn (str): the loss function used for training
        device (str): where the model and data should be loaded (gpu or cpu)
    
    Returns:
        None
    
    '''
    
    for epoch in range(1, epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        val_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # check if CUDA is available
            use_cuda = torch.cuda.is_available()
            # move to GPU if CUDA is available
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = loss_fn(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            # check if CUDA is available
            use_cuda = torch.cuda.is_available()
            # move tensors to GPU if CUDA is available
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = loss_fn(output, target)
            # update average validation loss 
            val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.data - val_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            val_loss
            ))


if __name__ == '__main__':
    '''Function used to set up an argument parser so all of the model 
    parameters and training parameters can be easily accessed
            
    Args:
        None
    
    Returns:
        None
    
    '''

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='L',
                        help='learning rate (default: 0.001)')

    # Dataloader Parameters
    parser.add_argument('--val-split', type=float, default=0.2, metavar='V',
                        help='split of the validation set (default: 0.2)')
    parser.add_argument('--data-mean', type=tuple, default=(0.4954160,), metavar='M',
                        help='mean of the dataset (default: (0.4954160,))')
    parser.add_argument('--data-std', type=tuple, default=(0.0564721,), metavar='S',
                        help='standard deviation of the dataset (default: (0.0564721,))')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}.'.format(device))

    torch.manual_seed(args.seed)

    # Load the training data
    train_loader = _get_data_loaders(args.batch_size, 
                                     args.data_dir, 
                                     args.data_mean, 
                                     args.data_std, 
                                     args.val_split)

    # Build the model
    model = CNNClassifier().to(device)

    # Train the model
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)