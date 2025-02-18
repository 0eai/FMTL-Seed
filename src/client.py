import os
import torch
import torch.nn as nn
import torch.optim as optim
import io
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
from collections import OrderedDict
from src.utils import Logger, create_datasets, deserialize_model, serialize_tensor_dict

class Client(object):
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.__id = client_id
        self.data = local_data
        self.device = device
        self.__model = None

    @property
    def id(self):
        return self.__id
    
    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, client_config):
        """Set up common configuration of each client; called by center server."""
        self.client_config = client_config
        
        self.dataloader = DataLoader(self.data, batch_size=client_config.batch_size, shuffle=True, num_workers=client_config.batch_size//4)
        
        self.local_epochs = client_config.local_epochs
        self.criterion = getattr(torch.nn, client_config.criterion.type)
        self.optimizer = getattr(torch.optim, client_config.optimizer.type)
        self.optim_config = client_config.optimizer.config
        self.criterion_config = client_config.criterion.config
        
        print('self.optimizer', type(self.optimizer), self.optimizer)
        print('self.criterion', type(self.criterion), self.criterion)
        
        print('self.optim_config', type(self.optim_config), self.optim_config)
        print('self.criterion_config', type(self.criterion_config), self.criterion_config)
        

    def client_update(self, round):
        self.model.train()
        self.model.to(self.device)

        optimizer = self.optimizer(self.model.parameters(), **self.optim_config.model_dump())
        print(f"\t[Client {str(self.id).zfill(4)} | Round {str(round).zfill(4)}] ...started training")
        for e in range(self.local_epochs):
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion()(outputs, labels)

                loss.backward()
                optimizer.step() 

                # if self.device.type == "cuda": torch.cuda.empty_cache()       
        print(f"\t[Client {str(self.id).zfill(4)} | Round {str(round).zfill(4)}] ...finished training")        
        self.model.to("cpu")

    def client_evaluate(self, round):
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            print(f"\t[Client {str(self.id).zfill(4)} | Round {str(round).zfill(4)}] ...started evaluation")
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss = test_loss + self.criterion()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()

                # if self.device.type == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)

        print(f"\t[Client {str(self.id).zfill(4)} | Round {str(round).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\n")

        # wandb.log({f'client_{self.id}_loss': test_loss, f'client_{self.id}_accuracy': test_accuracy}, step=round)
        return test_loss, test_accuracy

    def compute_model_update(self):
        model_update = OrderedDict()
        for key in self.model.state_dict().keys():
            model_update[key] = self.model.state_dict()[key] - self.initial_state_dict[key]
        return model_update

