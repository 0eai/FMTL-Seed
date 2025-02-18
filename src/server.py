import wandb
from datetime import datetime
from dateutil import tz
import torch
import torch.nn as nn
import io
import copy
import asyncio
from collections import OrderedDict
import random
from typing import Dict, List, Set, Optional
from fastapi import FastAPI, WebSocket

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split, Dataset

from src.models import TwoNN, ClientSystemInfo
from src.utils import Logger, create_datasets, serialize_model, deserialize_model, deserialize_tensor_dict


class Server(object):
    def __init__(self, config):
        self.connected_clients: Dict[str, WebSocket] = {}
        self.ready_clients: Set[str] = set()
        self.selected_clients = OrderedDict()
        self.client_system_info: Dict[str, ClientSystemInfo] = {}

        self.selected_clients_for_round: Set[str] = set()
        self.received_updates: Dict[str, dict] = {}  
        self._round = 0
        
        self.config = config
        
        # Global Config
        self.seed = config.seed

        # Data Config
        self.datasets = config.datasets

        # Server Config
        self.fraction = config.client_fraction
        self.num_clients = len(config.datasets)
        self.num_rounds = config.rounds
        self.batch_size = 64

        self.criterion = torch.nn.CrossEntropyLoss
        self.device = torch.device((config.device_type))
        self.test_datasets = dict()
        self.test_dataloaders = dict()
        
        self.global_model_lock = asyncio.Lock()  # For thread-safe model updates
        self.__model = None
        
    @property
    def model(self):
        """Server model getter."""
        return self.__model

    @model.setter
    def model(self, model):
        """Server model setter."""
        self.__model = model
    
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # split local dataset for each client
        for dataset_name in self.datasets:
            _, test_dataset = create_datasets(dataset_name=dataset_name, data_path='./data/')
            self.test_datasets[dataset_name] = test_dataset
            dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
            self.test_dataloaders[dataset_name] = dataloader
        
        self.model = TwoNN(name='TwoNN', in_features=784, num_hiddens=200, num_classes=10)
        print(f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")

    def sample_clients(self):
        """Select some fraction of all clients."""
        print(f"[Round: {str(self._round).zfill(4)}] Select clients...!")
        
        num_sampled_clients = max(int(self.fraction * len(self.ready_clients)), 1)
        selected_clients = random.sample(list(self.ready_clients), num_sampled_clients)
        self.selected_clients_for_round = set(selected_clients)
        print(f"[Round: {str(self._round).zfill(4)}] clients selected: {selected_clients}")
            
    def average_model(self, coefficients):
        """Apply the averaged updates to the global model."""
        print(f"[Round: {str(self._round).zfill(4)}] Aggregate updates from {len(self.selected_clients_for_round)} clients...!")

        try:
            averaged_state_dict = OrderedDict()
            for it, data in tqdm(enumerate(self.received_updates.values()), leave=False):
                model_update = data.model_state_dict
                for key, v in data.model_state_dict.items():
                    if isinstance(v, list):
                        v_tensor = torch.tensor(v)  # Convert list to tensor
                    else:
                        v_tensor = v  # Already a tensor, no need to convert.
                        
                    if it == 0:
                        averaged_state_dict[key] = coefficients[it] * v_tensor
                    else:
                        averaged_state_dict[key] += coefficients[it] * v_tensor
            
            # Apply the updates to the global model
            global_state_dict = self.model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] += averaged_state_dict[key]
        except Exception as e:
            print(e)
        print(f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(self.selected_clients_for_round)} clients are successfully averaged!")
        return global_state_dict

    async def aggregate_updates(self):
        # calculate averaging coefficient of weights
        total_data_size = sum(update.data_size for update in self.received_updates.values())
        mixing_coefficients = [update.data_size / total_data_size for update in self.received_updates.values()]

        # average each updated model parameters of the selected clients and update the global model
        global_state_dict = self.average_model(mixing_coefficients) # TODO Change name to FedAvg
        
        async with self.global_model_lock:
            global_state_dict = deserialize_model(global_state_dict)
            self.model.load_state_dict(global_state_dict)
        
    def evaluate_global_model(self, dataset, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += self.criterion()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
                
                if self.device.type == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(dataloader)
        test_accuracy = correct / len(dataset)
        return test_loss, test_accuracy
    
    def evaluate(self):
        for dataset_name in self.datasets:
            dataset = self.test_datasets[dataset_name]
            dataloader = self.test_dataloaders[dataset_name]
            
            test_loss, test_accuracy = self.evaluate_global_model(dataset, dataloader)
            wandb.log({f'{dataset_name}_global_loss': test_loss, f'{dataset_name}_global_accuracy': test_accuracy}, step=self._round)
                
            print(f"[Round: {str(self._round).zfill(4)}] [{dataset_name}] Loss: {test_loss:.4f} Accuracy: {100. * test_accuracy:.2f}%") 

