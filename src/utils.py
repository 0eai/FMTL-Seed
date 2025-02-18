import sys
import subprocess
import os
import io
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import yaml
import torch
import psutil
import random
import argparse
from collections import OrderedDict
from src.models import ClientSystemInfo, CPUInfo, GPUInfo

def parse_server_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='dilab2.ssu.ac.kr', help='host')
    parser.add_argument('--port', type=int, default=5001, help='port')
    parser.add_argument("-d", "--datasets", type=str, nargs='+', default=["MNIST", "EMNIST", "KMNIST", "QMNIST", "FakeData", "FakeData"], # , "FakeData", "FakeData"
                            help='Specify the datasets.')
    parser.add_argument("-p", "--project", type=str, default="MTL_CL_TASK", help = "Project name")
    parser.add_argument("-c", "--client_fraction", type=float, default=1.0, help = "client fraction")
    parser.add_argument("-s", "--start_round", type=int, default=0, help = "Start round")
    parser.add_argument("-r", "--rounds", type=int, help = "num rounds")
    parser.add_argument("-dt", "--device_type", type=str, help = "device type")
    parser.add_argument('--seed', type=int, default=101,
                        help='Specify the initial random seed (default: 101).')

    parser.add_argument("-ts", "--train_split", type=float, default=0.00, help = "train split")
    parser.add_argument("-set", "--train_set", type=str, default="train", help = "train_set")

    parser.add_argument("-b", "--backbone", type=str, default="resnet50", help = "Backbone")
    parser.add_argument("-wb", "--weights_backbone", type=str, default=None, help = "Backbone")
    parser.add_argument("-o", "--out_path", type=str, default="outs", help = "out_path")

    return parser.parse_args()

def parse_client_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='dilab2.ssu.ac.kr', help='host')
    parser.add_argument('--port', type=int, default=5001, help='port')
    parser.add_argument('--client_id', type=int, default=0, help='Client ID')
    parser.add_argument('--num_clients', type=int, default=2, help='Total number of clients')
    parser.add_argument('--seed', type=int, default=101,
                    help='Specify the initial random seed (default: 101).')
    return parser.parse_args()

def get_available_gpu_number(client_id, client_system_info, min_free_memory_mb=2048):
    client_sys_info = client_system_info[client_id]
    
    if not client_sys_info or not client_sys_info.cuda_available:
        return None  # No CUDA-enabled GPUs available or client data not found

    client_sys_id = client_sys_info.machine_id
    if not client_sys_id:
        print(f"Warning: IP address not found for client {client_id}.")
        return None
    
    # Build a mapping of GPUs to clients based on IP address
    gpu_to_client_map = {}  # {gpu_index: client_sys_id}
    for other_client_id, other_client_sys_info in client_system_info.items():
        if other_client_id != client_id and other_client_sys_info.cuda_available:
            other_client_sys_id = other_client_sys_info.machine_id
            assigned_gpu = other_client_sys_info.assigned_gpu
            if assigned_gpu is not None and other_client_sys_id == client_sys_id:
                gpu_to_client_map[assigned_gpu] = other_client_sys_id

    gpus = client_sys_info.gpus
    if not gpus:
        return None  # No GPU information found

    available_gpus = []
    for i, gpu in enumerate(gpus):
        try:
            free_memory_mb = float(gpu.free_memory_mb)
            # Check if GPU is already assigned to a client with the same IP
            if i not in gpu_to_client_map and free_memory_mb >= min_free_memory_mb:
                available_gpus.append((i, free_memory_mb))
        except (KeyError, ValueError, TypeError):
            print(f"Warning: Could not parse free memory for GPU {i} on client {client_id}.")
            continue

    if not available_gpus:
        return None  # No GPUs available based on the criteria

    # Sort by available memory in descending order
    available_gpus.sort(key=lambda x: x[1], reverse=True)

    # Choose the GPU with the most free memory
    selected_gpu_index = available_gpus[0][0]

    client_sys_info.assigned_gpu = selected_gpu_index
    return selected_gpu_index

def get_machine_id_linux():
    """Gets the machine ID on Linux systems."""
    # Method 1: Try using dbus (most reliable if available)
    try:
        result = subprocess.run(['dbus-send', '--system', '--print-reply', '--dest=org.freedesktop.machine1', '/org/freedesktop/machine1', 'org.freedesktop.DBus.Properties.Get', 'string:org.freedesktop.machine1', 'string:Id'], capture_output=True, text=True, check=True)
        # Extract the ID from the dbus output (it's a string)
        machine_id = result.stdout.strip().split('"')[1]
        return machine_id
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        pass  # Fallback to the next method

    # Method 2: Read from /etc/machine-id (less reliable, but a good fallback)
    try:
        with open('/etc/machine-id', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        pass
    
    #Method 3:  Read from /var/lib/dbus/machine-id
    try:
        with open('/var/lib/dbus/machine-id', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Unknown"
    
def get_system_info() -> ClientSystemInfo:
    """Collects system information."""
    print("Getting system info")
    # --- Memory Information ---
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024 ** 3)  # in GB
    available_memory = mem.available / (1024 ** 3)  # in GB

    # --- CPU Information ---
    cpu_info = CPUInfo(
        cpu_count=psutil.cpu_count(logical=True),  # Total logical CPUs
        cpu_physical_cores=psutil.cpu_count(logical=False),  # Physical cores
        cpu_frequency=psutil.cpu_freq().current,  # Current CPU frequency in MHz
        cpu_usage=psutil.cpu_percent(interval=1)  # CPU usage as a percentage
    )
        
    gpus = []
    if torch.cuda.is_available():
        cuda_available = True
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            try:
                # Get memory info in bytes and convert to MB
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                gpu_info = GPUInfo(
                    name=torch.cuda.get_device_name(i),
                    total_memory_mb=total_mem / (1024 ** 2),  # Convert to MB
                    free_memory_mb=free_mem / (1024 ** 2),    # Convert to MB
                    used_memory_mb=(total_mem - free_mem) / (1024 ** 2)  # Used memory in MB
                )
                gpus.append(gpu_info)
            except RuntimeError as e:
                print(f"Error getting memory info for GPU {i}: {e}")
                gpu_info = GPUInfo(name=torch.cuda.get_device_name(i)) # Keep only name.
                gpus.append(gpu_info)
    else:
        cuda_available = False
        gpu_count = 0

    return ClientSystemInfo(
        machine_id=get_machine_id_linux(),
        total_memory=total_memory,
        available_memory=available_memory,
        cpu=cpu_info,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpus=gpus
    )

class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()
        
def create_datasets(dataset_name='MNIST', data_path='./data/'):
    # Download and transform the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'EMNIST':
        train_dataset = datasets.EMNIST(root=data_path, train=True, download=True, split="digits", transform=transform)
        test_dataset = datasets.EMNIST(root=data_path, train=False, download=True, split="digits", transform=transform)
    elif dataset_name == 'KMNIST':
        train_dataset = datasets.KMNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'QMNIST':
        train_dataset = datasets.QMNIST(root=data_path, what="nist", train=True, download=True, transform=transform)
        test_dataset = datasets.QMNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'FakeData':
        train_dataset = datasets.FakeData(size=60000, image_size=(28, 28), num_classes=10, transform=transform, random_offset= random.randrange(0, 50, 10))
        test_dataset = datasets.FakeData(size=5000, image_size=(28, 28), num_classes=10, transform=transform, random_offset= random.randrange(0, 50, 10))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset


def serialize_model(model: torch.nn.Module) -> dict:
    return {k: v.tolist() for k, v in model.state_dict().items()}

def serialize_state_dict(state_dict: OrderedDict) -> dict:
    return {k: v.tolist() for k, v in state_dict.items()}

def deserialize_model(model_state_dict: dict):
    state_dict = {}
    for k, v in model_state_dict.items():
        if isinstance(v, list):
            # Convert lists back to tensors
            state_dict[k] = torch.tensor(v)
        else:
            # If it was already a tensor (shouldn't normally happen
            # with tolist() serialization, but good to be safe)
            state_dict[k] = v.clone().detach()  # Or just v if you are sure
    return state_dict

def serialize_tensor_dict(tensor_dict):
    buffer = io.BytesIO()
    torch.save(tensor_dict, buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_tensor_dict(data):
    buffer = io.BytesIO(data)
    return torch.load(buffer, weights_only=True)


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def read_yaml(file_path):
  """
  Reads a YAML file and returns its content as a Python object.

  Args:
    file_path: The path to the YAML file.

  Returns:
    A Python object with attributes corresponding to the YAML file's keys.
  """
  with open(file_path, 'r') as f:
    try:
      config_dict = yaml.safe_load(f)
      config_object = Config(**config_dict)
      return config_object
    except yaml.YAMLError as e:
      print(f"Error reading YAML file: {e}")
      return None
  
