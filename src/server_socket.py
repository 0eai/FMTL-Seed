import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import argparse
import wandb
import socketio
from collections import OrderedDict
import torch
import numpy as np
import random
from datetime import datetime
from dateutil import tz
from pathlib import Path

from src.server import Server
from src.utils import Logger, create_datasets, serialize_model, deserialize_model, deserialize_tensor_dict, read_yaml, get_available_gpu_number, Config


  
# Initialize Socket.IO server
sio = socketio.Server()
app = socketio.WSGIApp(sio)

# Event handlers
@sio.event
def connect(sid, environ):
    client_ip = environ.get('REMOTE_ADDR')
    forwarded_for = environ.get('HTTP_X_FORWARDED_FOR')

    if forwarded_for:
        # X-Forwarded-For might contain multiple IPs; take the first one
        client_ip = forwarded_for.split(',')[0].strip()
    elif environ.get('HTTP_X_REAL_IP'):
        client_ip = environ.get('HTTP_X_REAL_IP')

    print(f'Client connected: {sid} from {client_ip}')
    # Store the IP address temporarily, we'll merge it later in handle_join_request
    sio.save_session(sid, {'ip_address': client_ip}) 

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)
    if sid in server.clients:
        del server.clients[sid]
    if sid in server.selected_clients:
        del server.selected_clients[sid]

@sio.on('join_request')
def handle_join_request(sid, data):
    print('Client requested to join:', sid, data)

    # Retrieve the saved session data, including the IP address
    session_data = sio.get_session(sid)
    client_ip = session_data.get('ip_address')

    # Merge the IP address with the incoming data
    merged_data = data.copy()  # Create a copy to avoid modifying the original data
    merged_data['ip_address'] = client_ip

    # Store the merged data in server.clients
    server.clients[sid] = merged_data
    
    # Send acknowledgment
    # sio.emit('join_ack', {'message': 'Joined federation'}, room=sid)
        
    dataset = server.datasets[len(server.clients) -1]
    client_cfg = read_yaml(f'configs/{dataset.lower()}.yaml')
    client_cfg.project = args.project
    client_cfg.run_id = args.run_id
    client_cfg.out_path = args.out_path
    client_cfg.weight_path = f"{args.out_path}/clients/{dataset}/checkpoints/"
    client_cfg.log_path = f"{args.out_path}/clients/{dataset}/logs/"
    
    client_cfg.train_split = args.train_split
    client_cfg.train_set = args.train_set
    
    if data['cuda_available'] == True:
        gpu_number = get_available_gpu_number(server.clients, sid)
        if gpu_number is not None:
            client_cfg.device = f'cuda:{gpu_number}'
            print(f"Assigned GPU {gpu_number} to client {data['client_id']} (IP: {server.clients[sid]['ip_address']})")
        else:
            client_cfg.device = 'cpu'  # Fallback to CPU
            print(f"No suitable GPU found for client {data['client_id']} (IP: {server.clients[sid]['ip_address']}), using CPU.")
    else:
        client_cfg.device = 'cpu'
        print(f"Client {data['client_id']} does not have CUDA available, using CPU.")
    
    # Convert client_cfg to a dictionary if necessary
    if not isinstance(client_cfg, dict):
        client_cfg = vars(client_cfg)  # Assuming client_cfg is an object with attributes
    
    sio.emit('join_ack', data=client_cfg, room=sid)


@sio.on('client_setup')
def on_client_setup(sid, data):
    print('Client is ready for training:', sid, data)
    
    server.ready_clients[sid] = data['client_id']
    # Start training if enough clients have joined
    if len(server.clients) >= server.num_clients and len(server.ready_clients) >= server.num_clients and server._round == 0:
        sorted_items = sorted(server.clients.items(), key=lambda item: item[1]['client_id'])
        server.clients = OrderedDict(sorted_items)
        server.fit(sio)
    
@sio.on('client_update')
def handle_client_update(sid, data):
    print(f"[Server] Received update from client {sid} for round {server._round}")
    if sid in server.selected_clients:
        # Deserialize client model update
        client_model_update = deserialize_tensor_dict(data['model_update'])
        data['model_update'] = client_model_update
        # Move tensors to the appropriate device
        for key in client_model_update:
            client_model_update[key] = client_model_update[key].to(server.device)
        # Store the client update
        server.selected_clients[sid] = data
        
        wandb.log({f'client_{data["client_id"]}_loss': data["loss"], f'client_{data["client_id"]}_accuracy': data["accuracy"]}, step=data["round"])
        
        # If all selected clients have sent updates, aggregate
        if all(server.selected_clients.values()):
            server.aggregate_updates()
            server.evaluate()
            # Clear selected clients for next round
            server.selected_clients.clear()
            
            # Start next round
            server.fit(sio=sio)
    else:
        print(f"[Server] Ignoring update from unselected client {sid}")

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='dilab2.ssu.ac.kr', help='host')
parser.add_argument('--port', type=int, default=5001, help='port')
parser.add_argument("-d", "--datasets", type=str, nargs='+', default=["MNIST", "EMNIST", "KMNIST", "QMNIST", "EMNIST", "KMNIST"],
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

# parser.add_argument("-k", "--n_client", type=int, help = "number of clients")

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    ##############################################################
    args.run_id = '{}_{}_[{}]_[{}_{}]_[{}]_[{}_{}]_{}'.format('FL', datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.backbone,
                                                args.start_round, args.rounds, args.client_fraction, args.train_split, args.train_set, args.seed)
    out_path = Path(args.out_path) / args.project / args.run_id
    args.out_path = str(out_path)
    os.makedirs(args.out_path, exist_ok=True)
        
    args.weight_path = str(out_path / 'server' / 'checkpoints')
    args.log_path = str(out_path / 'server' / 'logs')
    os.makedirs(args.weight_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    wandb.init(entity='ssu', project=args.project, dir='wandb')
    print('wandb.config:', wandb.config)
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    wandb.run.name = args.run_id
    
    sys.stdout = Logger(os.path.join(args.log_path, args.run_id + '.txt'))
    print(' '.join(sys.argv))
    #################################################################################
    
    args.n_client= len(args.datasets)
    print("\n[WELCOME] Unfolding configurations...!")
    server = Server(args)
    server.setup()
    server.transmit_model(sio)
    
    
    import eventlet
    print("[Server] Starting server...")
    # eventlet.wsgi.server(eventlet.listen(('localhost', 5001)), app)
    
    # Start the server in a greenlet
    server_greenlet = eventlet.spawn(eventlet.wsgi.server, eventlet.listen((args.host, args.port)), app)
    
    # Main loop to monitor server status
    try:
        while not getattr(server, 'stop_server', False):
            eventlet.sleep(1)  # Yield control to allow server to process events
    except KeyboardInterrupt:
        print("\n[Server] Interrupt received. Shutting down...")
    finally:
        server_greenlet.kill()  # Stop the server greenlet
        print("[Server] Server stopped.")
    
    

# python src/server.py --dataset MNIST --n_client 4 --client_fraction 0.5 --rounds 10 --device_type cuda:3
# python src/client_socket.py --client_id 3 --num_clients 4
# export PYTHONPATH=$PYTHONPATH:/home/ankit/Projects/FMTL-Seed
# python src/server_socket.py --rounds 5 --device_type cpu


# TODO
# 1. Add gpu device available on client device
# 2. save client output(logs, checkpoints) and