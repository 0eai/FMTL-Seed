import os
import sys
from pathlib import Path
import asyncio
import json
import numpy as np
import random
from datetime import datetime
from dateutil import tz
import wandb
import torch
import signal
import uvloop
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from src.models import (
    JoinRequest, 
    ReadyMessage, 
    ModelUpdate, 
    ClientDefaultConfig, 
    CriterionConfig, 
    OptimizerConfig, 
    CrossEntropyLossConfig, 
    SGDOptimizerConfig
)
from src.server import Server
from src.utils import parse_server_args, Logger, read_yaml, get_available_gpu_number, serialize_model, deserialize_model

app = FastAPI()

def read_config(server, client_id):
    print("Reading client config")
    dataset = server.datasets[len(server.connected_clients) -1]
    client_cfg = read_yaml(f'configs/{dataset.lower()}.yaml')
        
    sys_info = server.client_system_info[client_id]
    
    if sys_info.cuda_available == True:
        gpu_number = get_available_gpu_number(client_id, server.client_system_info, client_cfg.min_free_memory)
        if gpu_number is not None:
            device = f'cuda:{gpu_number}'
            print(f"Assigned GPU {gpu_number} to client {client_id} (machine_id: {server.client_system_info[client_id].machine_id})")
        else:
            device = 'cpu'  # Fallback to CPU
            print(f"No suitable GPU found for client {client_id} (machine_id: {server.client_system_info[client_id].machine_id}), using CPU.")
    else:
        device = 'cpu'
        print(f"Client {client_id} does not have CUDA available, using CPU.")
    
    print('client_cfg.optim_config -> ', client_cfg.optim_config)
    criterion = CriterionConfig(
        type=client_cfg.criterion,
        config=CrossEntropyLossConfig(
            reduction=client_cfg.criterion_config[0]["reduction"],
            beta=client_cfg.criterion_config[0]["beta"]
        )
    )
    optimizer = OptimizerConfig(
        type=client_cfg.optimizer,
        config=SGDOptimizerConfig(
            lr=client_cfg.optim_config[0]["lr"],
            momentum=client_cfg.optim_config[0]["momentum"]
        )
    )
    
    return ClientDefaultConfig(
        project=server.config.project,
        run_id=server.config.run_id,
        out_path=server.config.out_path,
        weight_path=f"{server.config.out_path}/clients/{dataset}/checkpoints/",
        log_path=f"{server.config.out_path}/clients/{dataset}/logs/",
        train_split=server.config.train_split,
        train_set=server.config.train_set,
        device=device,
        local_epochs=client_cfg.local_epochs,
        batch_size=client_cfg.batch_size,
        dataset_name=client_cfg.dataset_name,
        criterion=criterion,
        optimizer=optimizer,
    )

# --- Checkpointing ---
CHECKPOINT_DIR = "checkpoints"  # Directory to store checkpoints
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "global_model.pth")

def save_checkpoint(model, round_num):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
    checkpoint = {
        'round': round_num,
        'model_state_dict': serialize_model(model),
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"Checkpoint saved to {CHECKPOINT_FILE}")

def load_checkpoint(model):
    global current_round  # Need to modify the global variable
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        state_dict = deserialize_model(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        current_round = checkpoint['round']
        print(f"Checkpoint loaded from {CHECKPOINT_FILE}, resuming from round {current_round}")
        return True
    else:
        print("No checkpoint found, starting from scratch.")
        return False

# --- WebSocket Endpoints ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    server.connected_clients[client_id] = websocket
    print(f"Client {client_id} connected. Total clients: {len(server.connected_clients)}")
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                print(f"Received message from {client_id}: {message["type"]}")
                
                if "type" not in message:
                    print(f"Ignoring the message from client {client_id}, message type not present")
                    continue
                
                if message["type"] == "join_request":
                    try:
                        join_request = JoinRequest(**message)
                        # Store client system information
                        server.client_system_info[join_request.client_id] = join_request.system_info
                        # Send config to the client

                        config = read_config(server, int(client_id))
                        await websocket.send_json({"type": "config", "data": config.model_dump()}) # Send as dict
                    except Exception as e:
                        print(f"An exception occured while parsing JoinRequest, {str(e)}")
                        await websocket.close(code=4000, reason=str(e)) #4000 custom code for parsing error
                        return
                
                elif message["type"] == "ready":
                    try:
                        ready_message = ReadyMessage.model_validate(message)
                        server.ready_clients.add(ready_message.client_id)
                        await handle_training_start()
                    except Exception as e:
                        print(f"An exception occured while parsing ReadyMessage, {str(e)}")
                        await websocket.close(code=4000, reason=str(e))
                        return  # Close and remove the connection.

                elif message["type"] == "model_update":
                    try:
                        update = ModelUpdate(**message)
                        wandb.log({f'client_{update.client_id}_loss': update.loss, f'client_{update.client_id}_accuracy': update.accuracy}, step=update.round)
                        server.received_updates[update.client_id] = update
                        await check_and_aggregate()
                    except Exception as e:
                        print(f"An exception occured while parsing ReadyMessage, {str(e)}")
                        await websocket.close(code=4000, reason=str(e))
                        return  # Close and remove the connection.
                else:
                    print(f"Ignoring the message from client {client_id}, message type: {message.get('type')} not supported")
            
            except json.JSONDecodeError as e:
                print(f"Invalid JSON from {client_id}: {e}")
                await websocket.close(code=4001, reason="Invalid JSON format")
                break
            except RuntimeError as e:
                if "WebSocket is not connected" in str(e):
                    print(f"Connection closed prematurely for {client_id}")
                    break
                raise

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    except Exception as e:
        print(f"Unexpected error with client {client_id}: {str(e)}")
    finally:
        print(f"Clean up on disconnect of client {client_id}")
        # Clean up on disconnect
        if client_id in server.connected_clients: del server.connected_clients[client_id]
        if client_id in server.ready_clients: server.ready_clients.remove(client_id)
        if client_id in server.client_system_info: del server.client_system_info[client_id]
        # If client disconnects during selection, remove it.
        if client_id in server.selected_clients_for_round: server.selected_clients_for_round.remove(client_id)
        if client_id in server.received_updates: del server.received_updates[client_id]

async def handle_training_start():
    global server
    if len(server.ready_clients) == server.num_clients and server._round < server.num_rounds:
        # All clients are ready, start a training round.
        server._round += 1
        print(f"[Round: {str(server._round).zfill(4)}] Starting training round {server._round}/{server.num_rounds}")
        server.sample_clients()
        server.received_updates.clear()  # Clear previous updates

        for client_id in server.selected_clients_for_round:
            websocket = server.connected_clients[str(client_id)]
            await websocket.send_json({"type": "start_training", "round": server._round})
            print(f"[Round: {str(server._round).zfill(4)}] sent start training to {client_id}")
        server.ready_clients.clear()


async def check_and_aggregate():
    global server

    if len(server.received_updates) == len(server.selected_clients_for_round):
        print(f"[Round: {str(server._round).zfill(4)}] Aggregating updates...")
        await server.aggregate_updates()
        server.evaluate()
        server.received_updates.clear() #Clear the updates
        server.selected_clients_for_round.clear() # Clear the selection
        print(f"[Round: {str(server._round).zfill(4)}] completed.")
        
        # Save checkpoint after each round
        save_checkpoint(server.model, server._round) #Save the checkpoint
        
        # Prepare for the next round or finish.
        if server._round < server.num_rounds:
            await asyncio.sleep(1)  # Give clients time to get ready
            # Start next round, server is contentiously listing for client messages
        else:
            await finish_training()


async def finish_training():
    print("Training completed. Sending finish message to all clients.")
    # Send finish message to all clients
    clients_copy = list(server.connected_clients.items())
    
    for client_id, websocket in clients_copy:
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "training_finished"})
                await websocket.close(code=1000)
                
        except Exception as e:
            print(f"Error closing connection with {client_id}: {e}")
    
    print("All rounds completed. Initiating server shutdown...")
    # Allow time for close frames to transmit
    await asyncio.sleep(2)
    
    del clients_copy
    
    # Graceful shutdown
    await shutdown_server()

async def shutdown_server():
    """Handles the graceful shutdown of the server."""
    print("Shutdown server")
    global server
    server.shutdown_flag = True

    # Close the uvicorn server
    server_instance = getattr(app, "server_instance", None)
    if server_instance:
        server_instance.should_exit = True
        # Give server time to handle existing requests
        await asyncio.sleep(1)

@app.get("/get_global_model")
async def get_global_model_endpoint():
    async with server.global_model_lock:
        serialized_model_data = serialize_model(server.model)
    return {"model_state_dict": serialized_model_data}

def main():
    global server
    
    args = parse_server_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
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
    
    # sys.stdout = Logger(os.path.join(args.log_path, args.run_id + '.txt'))
    print(' '.join(sys.argv))
    
    print("\n[WELCOME] Unfolding configurations...!")
    server = Server(args)
    server.setup()
    
    
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8008,
        ws_ping_interval=60,
        ws_ping_timeout=60,
        timeout_keep_alive=300,
        loop="uvloop"
    )
    server_instance = uvicorn.Server(config)
    app.server_instance = server_instance  # Store server instance reference
    server.shutdown_flag = False
    
    # Replace the signal handler setup with:
    async def shutdown():
        print("\nShutdown signal received")
        # Cancel running tasks except current
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        # Stop the server
        server_instance.should_exit = True

    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.add_signal_handler(signal.SIGINT, lambda: loop.create_task(shutdown()))
    loop.add_signal_handler(signal.SIGTERM, lambda: loop.create_task(shutdown()))
    
    try:
        loop.run_until_complete(server_instance.serve())
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        print("Cleaning up resources...")
        # Close wandb connection
        if wandb.run is not None:
            wandb.finish(exit_code=0)
        
        # Cleanup server resources
        # server.cleanup()  # Add cleanup method in Server class if needed
    
        # Handle async generators
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError:
            pass  # Already closed
        
        # Close the loop properly
        try:
            loop.close()
        except Exception as e:
            print(f"Error closing loop: {e}")
        print("Server shutdown complete")

if __name__ == "__main__":
    main()

# export PYTHONPATH=$PYTHONPATH:/home/ankit/Projects/FMTL-Seed
# python src/server_fastapi.py --host dilab2.ssu.ac.kr --port 5001 --rounds 5 --device_type cpu 