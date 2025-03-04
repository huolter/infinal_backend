from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Dict, List, Tuple
import asyncio
import random
import math
import noise
import os
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gualterio.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChunkGenerator:
    def __init__(self):
        self.chunks = {}
        self.DENSITY = 0.1
        self.seed = random.randint(0, 1000000)
        self.biome_scale = 100.0
        self.terrain_scale = 50.0

    def get_noise(self, x: float, z: float, scale: float) -> float:
        return noise.pnoise2(
            x / scale, z / scale, octaves=6, persistence=0.5, lacunarity=2.0,
            repeatx=1024, repeaty=1024, base=self.seed
        )

    def get_biome(self, chunk_x: int, chunk_z: int) -> str:
        biome_noise = self.get_noise(chunk_x * 16, chunk_z * 16, self.biome_scale)
        if biome_noise < -0.1:
            return 'desert'
        elif biome_noise < 0.1:
            return 'plains'
        else:
            return 'mountains'

    def generate_chunk(self, chunk_x: int, chunk_z: int) -> Dict:
        chunk_key = (chunk_x, chunk_z)
        if chunk_key in self.chunks:
            return self.chunks[chunk_key]

        biome = self.get_biome(chunk_x, chunk_z)
        chunk = {'terrain': [], 'features': [], 'entities': [], 'biome': biome}

        for z in range(16):
            terrain_row = []
            feature_row = []
            for x in range(16):
                wx = x + chunk_x * 16
                wz = z + chunk_z * 16
                height = self.get_noise(wx, wz, self.terrain_scale) * 10
                terrain_row.append(height)
                if random.random() < self.DENSITY:
                    feature_type = self.get_feature_type(wx, wz, biome)
                    feature_row.append(feature_type)
                else:
                    feature_row.append(0)
            chunk['terrain'].append(terrain_row)
            chunk['features'].append(feature_row)

        chunk['entities'] = self.generate_entities(chunk_x, chunk_z, biome)
        if biome == 'plains' and random.random() < 0.1:
            water_size = random.randint(3, 6)
            water_x = random.randint(0, 15 - water_size)
            water_z = random.randint(0, 15 - water_size)
            chunk['water'] = {'size': water_size, 'position': {'x': water_x, 'z': water_z}}

        self.chunks[chunk_key] = chunk
        return chunk

    def get_feature_type(self, x: float, z: float, biome: str) -> Dict:
        random.seed(int(x * 1000 + z * 1000 + self.seed))
        feature_value = random.random()
        size = random.uniform(0.5, 1.5)
        if biome == 'desert':
            if feature_value < 0.3:
                return {'type': 'cactus', 'size': size}
            elif feature_value < 0.6:
                return {'type': 'rock', 'size': size * 0.5}
            else:
                return {'type': 'dead_bush', 'size': size * 0.3}
        elif biome == 'plains':
            if feature_value < 0.2:
                return {'type': 'oak_tree', 'size': size}
            elif feature_value < 0.4:
                return {'type': 'birch_tree', 'size': size}
            elif feature_value < 0.6:
                return {'type': 'bush', 'size': size * 0.5}
            else:
                return {'type': 'flower', 'size': size * 0.2}
        else:  # mountains
            if feature_value < 0.4:
                return {'type': 'rock', 'size': size * 2}
            elif feature_value < 0.7:
                return {'type': 'pine_tree', 'size': size * 1.5}
            else:
                return {'type': 'snow_patch', 'size': size}
        return 0

    def generate_entities(self, chunk_x: int, chunk_z: int, biome: str) -> List[Dict]:
        entities = []
        if random.random() > 0.7:
            num_entities = random.randint(0, 2)
            for i in range(num_entities):
                entity_type = self.get_entity_type(biome)
                entities.append({
                    'type': entity_type,
                    'position': {'x': random.uniform(0, 16), 'y': 0, 'z': random.uniform(0, 16)},
                    'id': f"entity_{chunk_x}_{chunk_z}_{i}"
                })
        return entities

    def get_entity_type(self, biome: str) -> str:
        if biome == 'desert':
            return random.choice(['scorpion', 'snake', 'camel'])
        elif biome == 'plains':
            return random.choice(['deer', 'rabbit', 'fox'])
        else:  # mountains
            return random.choice(['eagle', 'goat', 'wolf'])

class GameState:
    def __init__(self):
        self.players: Dict[str, Dict] = {}
        self.chunk_generator = ChunkGenerator()
        self.VIEW_DISTANCE = 2
        self.time_of_day = 0
        self.DAY_NIGHT_CYCLE = 60

    def add_player(self, player_id: str, name: str = "Unnamed", position: Dict = None):
        if position is None:
            position = {'x': 0, 'y': 1.7, 'z': 0}
        self.players[player_id] = {
            'position': position,
            'rotation': 0,
            'active_chunks': set(),
            'name': name
        }

    def remove_player(self, player_id: str):
        if player_id in self.players:
            del self.players[player_id]

    def update_player_position(self, player_id: str, position: Dict, rotation: float):
        if player_id in self.players:
            self.players[player_id]['position'] = position
            self.players[player_id]['rotation'] = rotation

    def get_nearby_chunks(self, player_id: str) -> Dict[str, Dict]:
        if player_id not in self.players:
            return {}
        pos = self.players[player_id]['position']
        chunk_x = math.floor(pos['x'] / 16)
        chunk_z = math.floor(pos['z'] / 16)
        chunks = {}
        new_active_chunks = set()
        for dx in range(-self.VIEW_DISTANCE, self.VIEW_DISTANCE + 1):
            for dz in range(-self.VIEW_DISTANCE, self.VIEW_DISTANCE + 1):
                cx = chunk_x + dx
                cz = chunk_z + dz
                chunk_key = f"{cx},{cz}"
                new_active_chunks.add(chunk_key)
                chunks[chunk_key] = self.chunk_generator.generate_chunk(cx, cz)
        self.players[player_id]['active_chunks'] = new_active_chunks
        return chunks

    async def update_time(self, manager):
        while True:
            try:
                self.time_of_day = (self.time_of_day + 1 / (self.DAY_NIGHT_CYCLE * 10)) % 1
                await manager.broadcast({
                    'type': 'time_update',
                    'data': {'time_of_day': self.time_of_day}
                })
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in update_time: {e}")
                await asyncio.sleep(1)  # Back off on error

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.game_state = GameState()
        asyncio.create_task(self.game_state.update_time(self))
        asyncio.create_task(self.keepalive())  # Start keepalive task

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        player_id = str(len(self.active_connections))
        try:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message['type'] == 'set_name':
                name = message['data']['name']
            else:
                name = f"Player_{player_id}"
        except WebSocketDisconnect:
            name = f"Player_{player_id}"
            print(f"Client disconnected before sending name, assigned: {name}")
        self.game_state.add_player(player_id, name)
        chunks = self.game_state.get_nearby_chunks(player_id)
        
        players_data = {}
        for pid, player_data in self.game_state.players.items():
            player_data_copy = player_data.copy()
            player_data_copy['active_chunks'] = list(player_data_copy['active_chunks'])
            players_data[pid] = player_data_copy

        try:
            await websocket.send_text(json.dumps({
                'type': 'init',
                'data': {
                    'player_id': player_id,
                    'chunks': chunks,
                    'players': players_data,
                    'time_of_day': self.game_state.time_of_day
                }
            }))
            await self.broadcast({
                'type': 'player_joined',
                'data': {
                    'player_id': player_id,
                    'position': self.game_state.players[player_id]['position'],
                    'rotation': self.game_state.players[player_id]['rotation'],
                    'name': self.game_state.players[player_id]['name']
                }
            }, exclude=websocket)
        except WebSocketDisconnect:
            print(f"Client {player_id} disconnected during init")
            await self.disconnect(websocket, player_id)

    async def disconnect(self, websocket: WebSocket, player_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        self.game_state.remove_player(player_id)
        try:
            await self.broadcast({
                'type': 'player_left',
                'data': {'player_id': player_id}
            })
        except Exception as e:
            print(f"Error broadcasting player_left: {e}")

    async def broadcast(self, message: Dict, exclude: WebSocket = None):
        message_str = json.dumps(message)
        disconnected = []
        for connection in self.active_connections[:]:  # Copy to avoid modifying during iteration
            if connection != exclude:
                try:
                    await connection.send_text(message_str)
                except (WebSocketDisconnect, RuntimeError):
                    disconnected.append(connection)
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                print("Removed disconnected client from active_connections")

    async def update_player(self, websocket: WebSocket, player_id: str, data: Dict):
        position = data.get('position')
        rotation = data.get('rotation', 0)
        self.game_state.update_player_position(player_id, position, rotation)
        current_chunks = self.game_state.get_nearby_chunks(player_id)

        try:
            await websocket.send_text(json.dumps({
                'type': 'chunks_update',
                'data': {'chunks': current_chunks}
            }))
            print(f"Broadcasting position update for {player_id}: {position}")
            await self.broadcast({
                'type': 'position',
                'data': {
                    'player_id': player_id,
                    'position': position,
                    'rotation': rotation,
                    'name': self.game_state.players[player_id]['name']
                }
            }, exclude=websocket)
        except WebSocketDisconnect:
            print(f"Client {player_id} disconnected during update_player")
            await self.disconnect(websocket, player_id)

    async def keepalive(self):
        while True:
            await asyncio.sleep(30)  # Send keepalive every 30 seconds
            try:
                await self.broadcast({'type': 'ping'})
                print("Sent keepalive ping")
            except Exception as e:
                print(f"Error in keepalive: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    player_id = str(manager.active_connections.index(websocket) + 1)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message['type'] == 'position':
                await manager.update_player(websocket, player_id, message['data'])
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for player {player_id}")
        await manager.disconnect(websocket, player_id)
    except Exception as e:
        print(f"WebSocket error for player {player_id}: {e}")
        await manager.disconnect(websocket, player_id)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
