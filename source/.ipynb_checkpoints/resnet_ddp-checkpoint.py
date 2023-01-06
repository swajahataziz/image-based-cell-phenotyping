from __future__ import print_function

import argparse
import json
import logging
import os
from tqdm import tqdm

import numpy as np 

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import timm # PyTorch Image Models
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from torchvision import transforms as T,datasets

import glob
import cv2
import torch

import pickle
import random
import time
import copy
from torchsummary import  summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Import SMDataParallel PyTorch Modules
import smdistributed.dataparallel.torch.torch_smddp


# Use ResNet 50
import torchvision.models as models


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'tf_efficientnetv2_b1'
img_size = 244                          # Resize all the images to be 244 by 244
channels = 4
train_dir = '/opt/ml/input/data/training/'
test_dir = '/opt/ml/input/data/testing/'

class CustomDataset(Dataset):
    def __init__(self, root_dir, dim, channels, transform=None, TotalSamples=100):
        self.root_dir = root_dir
        self.transform = transform
        file_list = glob.glob(self.root_dir + "*")
        print(file_list)
        self.data = []
        self.datashape=(dim,dim,channels)
        self.TotalSamples=TotalSamples

        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for file_path in glob.glob(class_path + "/*.pickle"):
                #data.append([file_path, class_name])
                print(file_path)
                x = pickle.load(open(file_path,"rb"))
                x = tf.keras.utils.normalize(x)
                size = x.shape
                print(size)
                for image in range(size[0]):
                    im = [x[image].astype(float)]
                    im = np.array(im)
                    im = im.squeeze()  
                    if im.shape == self.datashape:
                        self.data.append([im, class_name])
        self.data = self.format_data(True)
        #print(self.data)

        self.class_map = {'HCT-116': 0, 'HL60': 1, 'JURKAT': 2, 'LNCAP': 3, 'MCF7': 4, 'PC3': 5, 'THP-1': 6, 'U2OS': 7}
        self.img_dim = (dim, dim)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, class_name = self.data[idx]
        #img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        #class_id = torch.tensor([class_id])
        #return img_tensor, class_id
        return img_tensor.float(), class_id
    
    def class_to_idx(self):
        return print(self.class_map)
    
    #Balance the classes so that they are of equal lengths
    #dataset is a list of [image, label]
    def format_data(self, augment):
        dataset = self.data
        classes = dict([])
        class_index = []
        data = []
        X = []
        y = []
        dataset_new=[]
        reverse_class_map = {0:'HCT-116' , 1:'HL60', 2:'JURKAT', 3:'LNCAP', 4:'MCF7', 5:'PC3', 6:'THP-1', 7:'U2OS'}
        for x in dataset:
            # check if exists in unique_list or not 
            if x[1] not in list(classes.keys()):
                classes[x[1]] = 1
            else:
                classes[x[1]] = classes[x[1]] + 1
            class_index.append(x[1])
            data.append(x[0])
        print(classes.items())

        if augment == True:
            for item in list(classes.keys()):
                indicies = [i for i, x in enumerate(class_index) if x == item] 
                if len(indicies) >= self.TotalSamples:
                    indicies = random.sample(indicies, k = self.TotalSamples)
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                else:
                    aug = []
                    for i in indicies:
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
                        #X.append(data[i])
                        #y.append(class_index[i])
                        aug.append(data[i])
                    new_data = self.data_augmentation(aug)
                    for i in range(len(new_data)):
                        #X.append(new_data[i])
                        #y.append(class_index[indicies[0]])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        else:
             for item in list(classes.keys()):
                    indicies = [i for i, x in enumerate(class_index) if x == item]
                    for i in indicies:
                        #X.append(data[i])
                        #y.append(class_index[i])
                        #class_name = reverse_class_map[class_index[i]]
                        dataset_new.append([data[i],class_index[i]])
        return dataset_new
    ##Rotational data augmentation
    def data_augmentation(self, data):
        new_data = []

        for i in range(self.TotalSamples-len(data)):
            new_image = data[random.randint(1,len(data)-1)]
            for r in range(random.randint(1,3)):
                new_image = np.rot90(new_image)
            new_data.append(new_image)
        return new_data

class CellPhenotypingTrainer():
    
    def __init__(self,criterion = None,optimizer = None,schedular = None):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
    
    def train_batch_loop(self,model,trainloader,args,epoch):
        
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (images, labels) in enumerate(trainloader):
        
        #for images,labels in tqdm(trainloader): 
            
            # move the data to CPU
            images = images.to(device)
            labels = labels.to(device)
            self.optimizer.zero_grad()
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(logits,labels)
            if batch_idx % args.log_interval == 0 and args.rank == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]; Train Loss: {:.6f}; Train Acc: {:.6f};".format(
                        epoch,
                        batch_idx * len(images) * args.world_size,
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        train_loss / len(trainloader),
                        train_acc / len(trainloader)
                    )
                )
            if args.verbose:
                print("Batch", batch_idx, "from rank", args.rank)            
            
        return train_loss / len(trainloader), train_acc / len(trainloader) 

    
    def valid_batch_loop(self,model,validloader):
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        with torch.no_grad():
            for images,labels in tqdm(validloader):

                # move the data to CPU
                images = images.to(device) 
                labels = labels.to(device)

                logits = model(images)
                loss = self.criterion(logits,labels)

                valid_loss += loss.item()
                valid_acc += accuracy(logits,labels)
            
        return valid_loss / len(validloader), valid_acc / len(validloader)
            
        
    def fit(self,model,trainloader,validloader,args,epochs):
        
        valid_min_loss = np.Inf 
        
        for i in range(epochs):
            
            model.train() # this turn on dropout
            avg_train_loss, avg_train_acc = self.train_batch_loop(model,trainloader,args,i) ###

            if args.rank == 0:
                model.eval()  # this turns off the dropout layer and batch norm
                avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model,validloader) ###
                if avg_valid_loss <= valid_min_loss :
                    print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                    torch.save(model.state_dict(),'/opt/ml/model/CellPhenotypingModel.pt')
                    valid_min_loss = avg_valid_loss
                print("Epoch : {} Valid Loss:{:.6f}; Valid Acc:{:.6f};".format(i+1, avg_valid_loss, avg_valid_acc))

            #print("Epoch : {} Train Loss:{:.6f}; Train Acc:{:.6f};".format(i+1, avg_train_loss, avg_train_acc))
            #print("Epoch : {} Valid Loss:{:.6f}; Valid Acc:{:.6f};".format(i+1, avg_valid_loss, avg_valid_acc))
            

def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def train_model(args):
    
    dist.init_process_group(backend="smddp")
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = int(os.getenv("LOCAL_RANK", -1))
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    
    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )

    if not torch.cuda.is_available():
        raise CUDANotFoundException(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )

    torch.manual_seed(args.seed)

    # select a single rank per node to download data
    is_first_local_rank = local_rank == 0
    if is_first_local_rank:
        trainset = CustomDataset(root_dir=train_dir, dim=244, channels=channels, TotalSamples=10000)

    dist.barrier()  # prevent other ranks from accessing the data early

    if not is_first_local_rank:
        trainset = CustomDataset(root_dir=train_dir, dim=244, channels=channels, TotalSamples=10000)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=args.world_size, rank=rank
    )
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True,sampler=train_sampler)
    
    #if rank == 0:
    testset = CustomDataset(root_dir=test_dir, dim=244, channels=channels, TotalSamples=1000)
    testloader = DataLoader(testset,batch_size=args.batch_size,shuffle=True)

    model = models.resnet50(pretrained=False)

    # modify first layer so it expects 4 input channels; all other parameters unchanged
    model.conv1 = torch.nn.Conv2d(4,64,kernel_size = (7,7),stride = (2,2), padding = (3,3), bias = False) 
    
    #we are updating it as a 8-class classifier:
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=625), #1280 is the orginal in_features
        nn.ReLU(), #ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=8), 
    )
    
    model = DDP(model.to(device), find_unused_parameters=True) # move the model to GPU
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    summary(model,input_size=(channels,img_size,img_size))


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)

    trainer = CellPhenotypingTrainer(criterion,optimizer)

    trainer.fit(model,trainloader,testloader,args,epochs = args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=40, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')

    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="For displaying smdistributed.dataparallel-specific logs")
    parser.add_argument('--hosts', type=json.loads, default=os.environ['SM_HOSTS'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    train_model(args=parser.parse_args())

def model_fn(model_dir):
    logger.info('model_fn')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_class = 8
    model = timm.create_model(model_name,pretrained=True) #load pretrained model
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1792, out_features=625), #1792 is the orginal in_features
        nn.ReLU(), #ReLu to be the activation function
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=8), 
        )
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)