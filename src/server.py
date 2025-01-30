import wandb
from datetime import datetime
from dateutil import tz
import torch
import torch.nn as nn
import io
import copy
from collections import OrderedDict
import random

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split, Dataset

from src.models import TwoNN
from src.utils import Logger, create_datasets, serialize_model, deserialize_model, deserialize_tensor_dict


class Server(object):
    def __init__(self, config):
        self.clients = OrderedDict()
        self.ready_clients = OrderedDict()
        self.selected_clients = OrderedDict()
        self._round = 0
        
        self.config = config
        # self.client_config = client_config
        
        # Global Config
        self.seed = config.seed

        # Data Config
        self.datasets = config.datasets

        # Server Config
        self.fraction = config.client_fraction
        self.num_clients = config.n_client
        self.num_rounds = config.rounds
        self.batch_size = 64

        self.criterion = torch.nn.CrossEntropyLoss
        self.device = torch.device((config.device_type))
        self.test_datasets = dict()
        self.test_dataloaders = dict()
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
            _, test_dataset = create_datasets(num_clients=self.num_clients, dataset_name=dataset_name, data_path='./data/')
            self.test_datasets[dataset_name] = test_dataset
            dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
            self.test_dataloaders[dataset_name] = dataloader
        
        self.model = TwoNN(name='TwoNN', in_features=784, num_hiddens=200, num_classes=10)
        print(f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!")
                
        # send the model skeleton to all clients
        # self.transmit_model()
    
    def transmit_model(self, sio):
        """Send the updated global model to selected/all clients."""
        # Send global model to selected clients
        model_state_dict = serialize_model(self.model)
        
        for sid in self.selected_clients.keys():
            sio.emit('model', data=model_state_dict, room=sid)
            print(f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted model to client {self.clients[sid]['client_id']}!")

    def sample_clients(self):
        """Select some fraction of all clients."""
        print(f"[Round: {str(self._round).zfill(4)}] Select clients...!")
        
        num_sampled_clients = max(int(self.fraction * len(self.clients)), 1)
        selected_sids = random.sample(list(self.clients.keys()), num_sampled_clients)
        self.selected_clients = {sid: None for sid in selected_sids}
        print('selected_sids ->', selected_sids)
            
    def update_selected_clients(self, sio):
        """Call "client_update" function of each selected client."""
        print(f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(self.selected_clients)} clients...!")
        
        for sid in self.selected_clients.keys():
            sio.emit('client_update', data={'round': self._round}, room=sid)
            
        print(f"[Round: {str(self._round).zfill(4)}] ...{len(self.selected_clients)} clients are selected and updated!")
    
    def average_model(self, coefficients):
        """Apply the averaged updates to the global model."""
        print(f"[Round: {str(self._round).zfill(4)}] Aggregate updates from {len(self.selected_clients)} clients...!")

        total_update = OrderedDict()
        for it, data in tqdm(enumerate(self.selected_clients.values()), leave=False):
            model_update = data['model_update']
            for key in self.model.state_dict().keys():
                if it == 0:
                    total_update[key] = coefficients[it] * model_update[key]
                else:
                    total_update[key] += coefficients[it] * model_update[key]
        # Apply the updates to the global model
        global_state_dict = self.model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] += total_update[key]
        self.model.load_state_dict(global_state_dict)
        
        print(f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(self.selected_clients)} clients are successfully averaged!")


    def evaluate_selected_models(self, sio):
        """Call "client_evaluate" function of each selected client."""
        print(f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(self.selected_clients))} clients' models...!")

        for sid in self.selected_clients.keys():
            sio.emit('client_evaluate', data={'round': self._round}, room=sid)

        print(f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(self.selected_clients))} selected clients!")
    
    def train_federated_model(self, sio):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sio)

        # updated selected clients with local dataset
        self.update_selected_clients(sio)
        
    def aggregate_updates(self):
        # calculate averaging coefficient of weights
        selected_total_size = sum([int(data['dataset_size']) for data in self.selected_clients.values()])
        mixing_coefficients = [int(data['dataset_size']) / selected_total_size for data in self.selected_clients.values()]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(mixing_coefficients) # TODO Change name to FedAvg
        
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
    
    def fit(self, sio):
        """Execute the whole process of the federated learning."""
        if self._round < self.num_rounds:
            self._round += 1
            self.train_federated_model(sio)
        else:
            print("\n[Server] Training complete")
            for sid in self.clients.keys():
                sio.emit('training_complete', room=sid)
            
            # Set a flag to stop the server
            self.stop_server = True
            
            
        # self.transmit_model()
        
    def evaluate(self):
        for dataset_name in self.datasets:
            dataset = self.test_datasets[dataset_name]
            dataloader = self.test_dataloaders[dataset_name]
            
            test_loss, test_accuracy = self.evaluate_global_model(dataset, dataloader)
            wandb.log({f'{dataset_name}_global_loss': test_loss, f'{dataset_name}_global_accuracy': test_accuracy}, step=self._round)
                
            print(f"[Round: {str(self._round).zfill(4)}] [{dataset_name}] Loss: {test_loss:.4f} Accuracy: {100. * test_accuracy:.2f}%") 

