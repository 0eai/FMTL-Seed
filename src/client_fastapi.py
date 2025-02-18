import argparse
import asyncio
import json
import httpx
import torch
from fastapi import HTTPException
import asyncio
import websockets
import json
import random
import numpy as np
from src.models import TwoNN, ReadyMessage, JoinRequest, ModelUpdate, ClientDefaultConfig
from src.utils import Logger, create_datasets, deserialize_model, serialize_state_dict, get_system_info, parse_client_args
from src.client import Client


# --- Helper functions ---
def generate_dummy_data(num_samples: int = 100):
    data = torch.randn(num_samples, 10)
    labels = torch.randint(0, 2, (num_samples,)).long()  # Binary classification
    return data, labels

# --- Client Class ---
class FederatedClient:
    def __init__(self, client_id: int, server_url: str):
        print("Initializing FederatedClient")
        self.server_url = server_url
        self.client_id = client_id # str(uuid.uuid4())
        self.websocket: websockets.WebSocketClientProtocol | None = None # Store the WebSocket

    async def connect(self):
        """Connects to the server and handles the main client logic."""
        print("Entering connect()")
        try:
            async with websockets.connect(f"{self.server_url}/{self.client_id}", ping_interval=60, ping_timeout=60) as websocket:
                self.websocket = websocket
                print("Connected to server")
                await self.send_join_request()
                await self.listen_for_server_messages()
        except ConnectionRefusedError:
            print(f"Failed to connect to the server at {self.server_url}.  Is the server running?")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:  # Ensure the connection is closed, even on errors.
            if self.websocket:
                await self.websocket.close()
                print("WebSocket connection closed.")

    async def send_join_request(self):
        """Sends a join request to the server."""
        print("Sending join request")
        system_info = get_system_info()
        join_request = JoinRequest(client_id=self.client_id, system_info=system_info)
        await self.websocket.send(json.dumps({"type": "join_request", **json.loads(join_request.model_dump_json())}))
        print("Join request sent.")


    async def listen_for_server_messages(self):
        """Listens for messages from the server and calls the appropriate handler."""
        print("Listening for server messages")
        try:
            async for message_str in self.websocket:
                message = json.loads(message_str)
                print(f"Received from server: {message}")
                if message["type"] == "config":
                    self.config = ClientDefaultConfig.model_validate(message["data"])
                    
                    await self.setup_client()
                    await self.send_ready_message()
                elif message["type"] == "start_training":
                    self.round = message["round"]
                    await self.train_and_send_update(self.round)
                elif message["type"] == "training_finished":
                    print("Training finished. Exiting.")
                    await self.websocket.close()
                    await asyncio.sleep(2)
                    return  # Exit the listening loop.

            else:
                print(f"Unknown message type: {message['type']}")
        except Exception as e:
            print("Exception in listen for server message ", e)
            return #Exit the listen loop.

    async def send_ready_message(self):
        """Sends a 'ready' message to the server."""
        print("Sending ready message")
        ready_message = ReadyMessage(client_id=self.client_id)
        await self.websocket.send(json.dumps({"type": "ready", **json.loads(ready_message.model_dump_json())}))
        print("Ready message sent.")


    async def train_and_send_update(self, round):
        """Downloads the global model, trains locally, and sends the update."""
        #Get Global Model
        print("Getting global model")
        await self.get_global_model()
        # Train
        loop = asyncio.get_event_loop()
        print(f"[Client {str(self.client.id).zfill(4)} | Round {str(round).zfill(4)}] Starting local training...")
        # await self.client.client_update(round=round)
        await loop.run_in_executor(None, self.client.client_update, round)
        print(f'[Client {str(self.client.id).zfill(4)} | Round {str(round).zfill(4)}] Local training complete')
        # test_loss, test_accuracy = await self.client.client_evaluate(round=round)
        test_loss, test_accuracy = await loop.run_in_executor(None, self.client.client_evaluate, round)
        print(f'[Client {str(self.client.id).zfill(4)} | Round {str(round).zfill(4)}] Local evaluation complete')
        
        await self.send_model_update(loss=test_loss, accuracy=test_accuracy)
        await self.send_ready_message()



    async def get_global_model(self):
        print("Requesting global model")
        async with httpx.AsyncClient() as client:
            url = f"http://localhost:8008/get_global_model"  
            response = await client.get(url)
            if response.status_code == 200:
                model_data = response.json()
                state_dict = deserialize_model(model_data['model_state_dict'])
                self.client.model.load_state_dict(state_dict)
                self.client.initial_state_dict = {key: value.clone() for key, value in state_dict.items()}
                print("Global model downloaded and loaded successfully.")
            else:
                print(f"Error downloading global model. Status code: {response.status_code}")
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch global model")

    async def send_model_update(self, loss: float, accuracy: float=None):
        """Sends the model update to the server."""
        print("Sending model update")
        serialized_update = serialize_state_dict(self.client.compute_model_update())
        model_update = ModelUpdate(
            client_id=str(self.client_id), 
            round=self.round, 
            model_state_dict=serialized_update, 
            data_size=len(self.client),
            loss=loss,
            accuracy=accuracy
        )
        await self.websocket.send(json.dumps({"type": "model_update", **json.loads(model_update.model_dump_json())}))
        print("Model update sent.")


    async def setup_client(self) -> None:
        train_dataset, test_dataset = create_datasets(dataset_name=self.config.dataset_name)
        
        device = torch.device(self.config.device)
        self.client = Client(client_id=self.client_id, local_data=train_dataset, device=device)
        self.client.model = TwoNN()
        self.client.setup(client_config=self.config)
        print("Client setup completed.")
        
# --- Main Execution (Client) ---
    
async def main_client():
    args = parse_client_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    print("Staring main client")
    client = FederatedClient(args.client_id ,server_url=f"ws://{args.host}:{args.port}/ws")
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main_client())
    
# export PYTHONPATH=$PYTHONPATH:/home/ankit/Projects/FMTL-Seed
# python src/client_fastapi.py --client_id 0 --num_clients 4 --host dilab2.ssu.ac.kr --port 8008