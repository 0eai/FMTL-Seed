import torch.nn as nn
from pydantic import BaseModel, ValidationError, field_validator
from typing import Literal, Optional, Union, Dict, Any, List, Set, Optional

class TwoNN(nn.Module):
    def __init__(self, name='TwoNN', in_features=784, num_hiddens=200, num_classes=10):
        super(TwoNN, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
        self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
        self.fc3 = nn.Linear(in_features=num_hiddens, out_features=num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x
    
# --- Pydantic Models (for data validation) ---

# --- Loss Function Configuration ---

class CrossEntropyLossConfig(BaseModel):
    reduction: Literal['none', 'mean', 'sum'] = 'mean'  # Default to 'mean'
    beta: Optional[float] = None # Added optional

class CriterionConfig(BaseModel):
    # Allow for different loss functions by using a type and config
    type: Literal['CrossEntropyLoss']  # Add other loss types here as needed
    config: CrossEntropyLossConfig


# --- Optimizer Configuration ---

class SGDOptimizerConfig(BaseModel):
    lr: float
    momentum: float = 0.0  # Make momentum optional with a default
    weight_decay: float = 0.0 # Added weight_decay
    dampening: float = 0.0 # Added dampening
    nesterov: bool = False # Added nesterov

class AdamOptimizerConfig(BaseModel):  # Added Adam config
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    
class OptimizerConfig(BaseModel):
    type: Literal['SGD', 'Adam']  # Allow different optimizer types
    config: Union[SGDOptimizerConfig, AdamOptimizerConfig] #Union is important
    
class ClientDefaultConfig(BaseModel):  # Configuration Model
    project: str
    run_id: str
    out_path: str
    weight_path: str
    log_path: str
    train_split: float
    train_set: str
    device: str
    local_epochs: int
    batch_size: int
    criterion: CriterionConfig
    optimizer: OptimizerConfig
    dataset_name: str
    
class GPUInfo(BaseModel):
    name: str
    total_memory_mb: float
    free_memory_mb: float
    used_memory_mb: float
    
class CPUInfo(BaseModel):
    cpu_count: int
    cpu_physical_cores: int
    cpu_frequency: float  # Changed to float
    cpu_usage: float      # Changed to float

class ClientSystemInfo(BaseModel):
    machine_id: str
    total_memory: float  # Changed to float
    available_memory: float  # Changed to float
    cpu: CPUInfo
    cuda_available: bool
    gpu_count: Optional[int] = 0
    assigned_gpu: Optional[int] = 0
    gpus: Optional[List[GPUInfo]] = None  # Made Optional

    @field_validator('gpus', mode='before')  # Use field_validator, mode='before'
    def set_gpus(cls, v, info):
        if info.data.get('cuda_available') and v is None:
            return []
        return v

class JoinRequest(BaseModel):
    client_id: int
    system_info: ClientSystemInfo

class ReadyMessage(BaseModel):
    client_id: int

class ModelUpdate(BaseModel):
    client_id: str
    round: int
    model_state_dict: dict
    data_size: int
    loss: float
    accuracy: float
