import os
import sys
import wandb
import socketio
import torch
import argparse
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.models import TwoNN
from src.utils import Logger, create_datasets, deserialize_model, serialize_tensor_dict, Config, get_system_resources
from src.client import Client

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.empty_cache()

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

executor = ThreadPoolExecutor(max_workers=1)

# Initialize Socket.IO client
sio = socketio.Client()

@sio.event
def connect():
    print(f'[Client({client_id})] Connected to server')
    data = get_system_resources()
    data["client_id"] = client_id
    # Request to join federation
    sio.emit('join_request', data=data)

@sio.event
def disconnect(reason=None):
    print(f'[Client({client_id})] Disconnected: {reason}')
    
@sio.on('join_ack')
def on_join_ack(data):
    global client_cfg
    print(f'[Client({client_id})] Received acknowledgment from server:', data)
    
    try:
        assert 'dataset_name' in data and 'min_free_memory' in data
        client_cfg = Config(**data)
    except (AssertionError, TypeError) as e:
        print(f"[Client({client_id})] Invalid join_ack data: {e}")
        sio.disconnect()
        
    client_setup(client_cfg)
    os.makedirs(client_cfg.weight_path, exist_ok=True)
    os.makedirs(client_cfg.log_path, exist_ok=True)
    

def client_setup(client_cfg):
    try:
        global client
        train_dataset, test_dataset = create_datasets(num_clients=num_clients, dataset_name=client_cfg.dataset_name)
        
        #torch.cuda.set_device(0)
        device = torch.device(client_cfg.device)
        client = Client(client_id=client_id, local_data=train_dataset, device=device)
        client.model = TwoNN()
        client.setup(client_config=client_cfg)
        
        sio.emit('client_setup', data={"client_id": client_id})
        print(f"[Client({client.id})] ✅ Client setup completed.")
    except Exception as e:
        print(f"[Client({client_id})] ❌ Setup failed: {e}")
        sio.emit('client_error', {'client_id': client_id, 'error': str(e)})

@sio.on('model')
def on_model(data, ack=None):
    """Acknowledge model receipt after validation."""
    print(f'[Client({client.id})] Received model from server')
    try:
        # Deserialize the model
        print(f'[Client({client.id})] Deserializing model')
        state_dict = deserialize_model(data)
        print(f'[Client({client.id})] Loading model state_dict')
        client.model.load_state_dict(state_dict)
        # Store the initial state_dict for computing updates
        client.initial_state_dict = {key: value.clone() for key, value in state_dict.items()}
        
        print(f"[Client({client.id})] ✅ Model loaded successfully.")
        if ack:  # Send ACK only if the server expects it
            ack()
    except Exception as e:
        print(f"[Client({client.id})] Model load failed: {e}")
        if ack:  # Check if acknowledgment is expected
            ack({"error": str(e)})
        else:
            sio.emit('client_error', {'client_id': client.id, 'error': str(e)})

@sio.on('client_update')
def on_client_update(data):
    def _process_update():
        try:
            round = data['round']
            client.client_update(round=round)
            print(f'[Client({client.id})] Local training complete')
            test_loss, test_accuracy = client.client_evaluate(round=round)
            print(f'[Client({client.id})] Local evaluation complete')

            # Compute model update
            model_update = client.compute_model_update()
            # Ensure tensors are on CPU before serialization
            for key in model_update:
                model_update[key] = model_update[key].to('cpu')
            # Serialize model update
            update_data = serialize_tensor_dict(model_update)
            
            payload = {
                'client_id': client.id,
                'round': round,
                'dataset_size': len(client),
                'model_update': update_data,
                'loss': test_loss,
                'accuracy': test_accuracy,
            }
            
            sio.emit('client_update', data=payload)
            print(f'[Client({client.id})] ✅ Sent updated model to server')
        except Exception as e:
            print(f"[Client({client.id})] ❌ Error in client_update: {e}")
            sio.emit('client_error', {'client_id': client.id, 'error': str(e)})
    executor.submit(_process_update)

@sio.on('client_evaluate') 
def on_client_evaluate(data):
    try:
        round = data['round']
        test_loss, test_accuracy = client.client_evaluate(round=round)
        print(f'[Client({client.id})] Local evaluation complete')
        payload = {
            'client_id': client.id,
            'round': round,
            'loss': test_loss,
            'accuracy': test_accuracy,
        }
        sio.emit('client_evaluate', data=payload)
        print(f'[Client({client.id})] ✅ Sent eval metrices to server')
    except Exception as e:
        print(f"[Client({client.id})] ❌ Error in client_update: {e}")
        sio.emit('client_error', {'client_id': client.id, 'error': str(e)})

@sio.on('training_complete')
def on_training_complete():
    print(f'[Client({client.id})] ✅ Training complete. Disconnecting...')
    # Perform any cleanup if necessary
    sio.disconnect()
    
        
if __name__ == '__main__':
    # import multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='dilab2.ssu.ac.kr', help='host')
    parser.add_argument('--port', type=int, default=5001, help='port')
    parser.add_argument('--client_id', type=int, default=0, help='Client ID')
    parser.add_argument('--num_clients', type=int, default=2, help='Total number of clients')
    parser.add_argument('--seed', type=int, default=101,
                    help='Specify the initial random seed (default: 101).')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    client_id = args.client_id
    num_clients = args.num_clients
    # device = torch.device('cpu')

    # Connect to server
    # sio.connect('http://localhost:5001')
    # sio.wait()
    # print(f'[Client({args.client_id})] Client has disconnected. Exiting...')

    # Connect to server
    try:
        sio.connect(f'http://{args.host}:{args.port}', wait_timeout=10, transports=['websocket'])
        sio.wait()  # Use built-in event loop
    except socketio.exceptions.ConnectionError:
        print(f'[Client({client_id})] Connection to server failed.')
    finally:
        print(f'[Client({client_id})] Client has disconnected. Exiting...')
        sys.exit(0)