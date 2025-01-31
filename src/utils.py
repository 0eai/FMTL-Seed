import sys
import io
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import yaml
import torch
import psutil

def get_available_gpu_number(server_clients, sid, min_free_memory_mb=2048):
    """
    Determines an available GPU number for a client based on GPU availability, 
    usage, and a one-client-per-GPU policy based on IP addresses.

    Args:
        server_clients (dict): The dictionary containing all client data.
        sid (str): The Socket.IO session ID of the current client.
        min_free_memory_mb (int): The minimum amount of free memory (in MB) 
                                 required for a GPU to be considered available.

    Returns:
        int or None: The index of an available GPU (starting from 0) if one is found, 
                     otherwise None (indicating no GPU is available or suitable).
    """
    client_data = server_clients.get(sid)
    
    if not client_data or not client_data.get('cuda_available', False):
        return None  # No CUDA-enabled GPUs available or client data not found

    client_ip = client_data.get('ip_address')
    if not client_ip:
        print(f"Warning: IP address not found for client {client_data['client_id']}.")
        return None
    
    # Build a mapping of GPUs to clients based on IP address
    gpu_to_client_map = {}  # {gpu_index: client_ip}
    for other_sid, other_client_data in server_clients.items():
        if other_sid != sid and other_client_data.get('cuda_available', False):
            other_client_ip = other_client_data.get('ip_address')
            assigned_gpu = other_client_data.get('assigned_gpu')
            if assigned_gpu is not None and other_client_ip == client_ip:
                gpu_to_client_map[assigned_gpu] = other_client_ip

    gpus = client_data.get('gpus', [])
    if not gpus:
        return None  # No GPU information found

    available_gpus = []
    for i, gpu in enumerate(gpus):
        try:
            free_memory_mb = float(gpu['free_memory_mb'])
            # Check if GPU is already assigned to a client with the same IP
            if i not in gpu_to_client_map and free_memory_mb >= min_free_memory_mb:
                available_gpus.append((i, free_memory_mb))
        except (KeyError, ValueError, TypeError):
            print(f"Warning: Could not parse free memory for GPU {i} on client {client_data['client_id']}.")
            continue

    if not available_gpus:
        return None  # No GPUs available based on the criteria

    # Sort by available memory in descending order
    available_gpus.sort(key=lambda x: x[1], reverse=True)

    # Choose the GPU with the most free memory
    selected_gpu_index = available_gpus[0][0]

    # Assign the selected GPU to the client
    client_data['assigned_gpu'] = selected_gpu_index
    return selected_gpu_index

def get_system_resources():
    """
    Gets available memory, CPU, and GPU information.

    Returns:
        dict: A dictionary containing memory, CPU, and GPU details.
    """

    resources = {}

    # --- Memory Information ---
    mem = psutil.virtual_memory()
    resources['total_memory'] = mem.total / (1024 ** 3)  # in GB
    resources['available_memory'] = mem.available / (1024 ** 3)  # in GB

    # --- CPU Information ---
    cpu_info = {
        'cpu_count': psutil.cpu_count(logical=True),  # Total logical CPUs
        'cpu_physical_cores': psutil.cpu_count(logical=False),  # Physical cores
        'cpu_frequency': psutil.cpu_freq().current,  # Current CPU frequency in MHz
        'cpu_usage': psutil.cpu_percent(interval=1)  # CPU usage as a percentage
    }
    resources['cpu'] = cpu_info

    # --- GPU Information ---
    if torch.cuda.is_available():
        resources['cuda_available'] = True
        resources['gpu_count'] = torch.cuda.device_count()
        gpus = []
        for i in range(resources['gpu_count']):
            try:
                # Get memory info in bytes and convert to MB
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_mb': total_mem / (1024 ** 2),  # Convert to MB
                    'free_memory_mb': free_mem / (1024 ** 2),    # Convert to MB
                    'used_memory_mb': (total_mem - free_mem) / (1024 ** 2)  # Used memory in MB
                }
                gpus.append(gpu_info)
            except RuntimeError as e:
                print(f"Error getting memory info for GPU {i}: {e}")
                gpu_info = {
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_mb': 'Unavailable',
                    'free_memory_mb': 'Unavailable',
                    'used_memory_mb': 'Unavailable'
                }
                gpus.append(gpu_info)
        resources['gpus'] = gpus
    else:
        resources['cuda_available'] = False

    return resources

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
        
def create_datasets(num_clients=10, dataset_name='MNIST', data_path='./data/'):
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
        train_dataset = datasets.KMNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.KMNIST(root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'FakeData':
        train_dataset = datasets.FakeData(size=60000, image_size=(28, 28), num_classes=10, transform=transform)
        test_dataset = datasets.FakeData(size=5000, image_size=(28, 28), num_classes=10, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, test_dataset

    # Split the training data into `num_clients` local datasets
    # total_samples = len(train_dataset)
    # client_split_sizes = [total_samples // num_clients] * (num_clients - 1)
    # client_split_sizes.append(total_samples - sum(client_split_sizes))  # Last client gets the remaining samples
    # client_datasets = random_split(train_dataset, client_split_sizes)

    # return client_datasets, test_dataset

def serialize_model(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_model(data):
    buffer = io.BytesIO(data)
    state_dict = torch.load(buffer, weights_only=True)
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
  
