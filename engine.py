from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Dict, Set, Optional, List
import asyncio
import random
import math
import noise
import os
import time
import zlib
from collections import OrderedDict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gualterio.com", "http://localhost:*", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LRUCache(OrderedDict):
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        super().__init__()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

class ChunkGenerator:
    def __init__(self):
        self.chunks = LRUCache(maxsize=2000)
        self.DENSITY = 0.3
        self.seed = random.randint(0, 1000000)
        self.biome_scale = 100.0
        self.terrain_scale = 50.0
        self.generated_chunks = 0
        self.cache_hits = 0

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
            self.cache_hits += 1
            return self.chunks[chunk_key]

        self.generated_chunks += 1

        biome = self.get_biome(chunk_x, chunk_z)
        chunk = {'terrain': [], 'features': [], 'biome': biome}

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

        chunk['static_entities'] = self.generate_static_entities(chunk_x, chunk_z, biome)

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

    def generate_static_entities(self, chunk_x: int, chunk_z: int, biome: str) -> list[dict]:
        entities = []
        if random.random() > 0.7:
            num_entities = random.randint(0, 2)
            for i in range(num_entities):
                entity_type = self.get_entity_type(biome)
                entities.append({
                    'type': entity_type,
                    'position': {'x': random.uniform(0, 16), 'y': 0, 'z': random.uniform(0, 16)},
                    'id': f"static_entity_{chunk_x}_{chunk_z}_{i}"
                })
        return entities

    def get_entity_type(self, biome: str) -> str:
        if biome == 'desert':
            return random.choice(['scorpion', 'snake', 'camel'])
        elif biome == 'plains':
            return random.choice(['deer', 'rabbit', 'fox'])
        else:  # mountains
            return random.choice(['eagle', 'goat', 'wolf'])

    def get_stats(self) -> Dict:
        return {
            "total_chunks_generated": self.generated_chunks,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.chunks)
        }

class NPC:
    FIRST_NAMES = ["zappa", "damePapusa", "jive", "rock", "jazz", "punk", "JUanDomingo", "flare", "nova", "pulse"]
    SECOND_NAMES = ["narcisus", "jupiter", "TRUMP", "mars", "venus", "apollo", "Peron", "athena", "hermes", "artemis"]
    
    def __init__(self, npc_id: str):
        self.id = npc_id
        self.name = self.generate_name()
        self.position = {
            'x': random.uniform(-100, 100),
            'y': random.uniform(0, 10),
            'z': random.uniform(-100, 100)
        }
        self.rotation = random.uniform(0, 2 * math.pi)
        self.size = random.uniform(1, 25)
        self.scale_factor = 1.0  # Track current scale relative to original size
        self.speed = random.uniform(0.02, 0.1)
        self.direction = {
            'x': math.cos(self.rotation) * self.speed,
            'z': math.sin(self.rotation) * self.speed
        }
        
    def generate_name(self) -> str:
        first = random.choice(self.FIRST_NAMES)
        number = random.randint(1, 99)
        return f"{first}_{number}"
    
    def update_position(self):
        self.position['x'] += self.direction['x']
        self.position['z'] += self.direction['z']
        
        if random.random() < 0.005 or abs(self.position['x']) > 500 or abs(self.position['z']) > 500:
            new_rotation = random.uniform(0, 2 * math.pi)
            self.rotation = new_rotation
            self.direction = {
                'x': math.cos(self.rotation) * self.speed,
                'z': math.sin(self.rotation) * self.speed
            }
            if abs(self.position['x']) > 500 or abs(self.position['z']) > 500:
                angle_to_center = math.atan2(-self.position['z'], -self.position['x'])
                self.rotation = angle_to_center
                self.direction = {
                    'x': math.cos(self.rotation) * self.speed,
                    'z': math.sin(self.rotation) * self.speed
                }
    
    def hit(self):
        self.scale_factor = max(self.scale_factor * 0.9, 0.1)  # Shrink by 10%, min 0.1

    def get_data(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'position': self.position,
            'rotation': self.rotation,
            'size': self.size * self.scale_factor  # Adjusted size based on scale factor
        }

class GameState:
    def __init__(self, num_npcs: int = 20):
        self.players: Dict[str, Dict] = {}
        self.npcs: Dict[str, NPC] = {}
        self.bullets = {}
        self.chunk_generator = ChunkGenerator()
        self.VIEW_DISTANCE = 3
        self.time_of_day = 0
        self.DAY_NIGHT_CYCLE = 240
        self.last_activity = {}
        self.connections_total = 0
        self.connections_active = 0
        self.messages_received = 0
        self.messages_sent = 0
        
        self.init_npcs(num_npcs)
        
    def init_npcs(self, num_npcs: int):
        for i in range(num_npcs):
            npc_id = f"npc_{i}"
            self.npcs[npc_id] = NPC(npc_id)
            
    def update_npcs(self):
        for npc in self.npcs.values():
            npc.update_position()
            
    def get_npcs_data(self) -> Dict[str, Dict]:
        return {npc_id: npc.get_data() for npc_id, npc in self.npcs.items()}

    def add_player(self, player_id: str, name: str = "Unnamed", position: Dict = None):
        if position is None:
            position = {'x': 0, 'y': 1.7, 'z': 0}
        self.players[player_id] = {
            'position': position,
            'rotation': 0,
            'active_chunks': set(),
            'name': name,
            'joined_at': time.time(),
            'scale_factor': 1.0  # Track player scale
        }
        self.last_activity[player_id] = time.time()
        self.connections_total += 1
        self.connections_active += 1

    def remove_player(self, player_id: str):
        if player_id in self.players:
            del self.players[player_id]
        if player_id in self.last_activity:
            del self.last_activity[player_id]
        self.connections_active -= 1

    def update_player_position(self, player_id: str, position: Dict, rotation: float):
        if player_id in self.players:
            self.players[player_id]['position'] = position
            self.players[player_id]['rotation'] = rotation
            self.last_activity[player_id] = time.time()

    def add_bullet(self, bullet_id: str, position: Dict, direction: Dict, shooter_id: str):
        self.bullets[bullet_id] = {
            'position': position,
            'direction': direction,
            'shooter_id': shooter_id,
            'lifetime': 3.0
        }

    def update_bullets(self, delta_time: float):
        expired = []
        for bullet_id, bullet in list(self.bullets.items()):
            bullet['position']['x'] += bullet['direction']['x'] * 0.5
            bullet['position']['y'] += bullet['direction']['y'] * 0.5
            bullet['position']['z'] += bullet['direction']['z'] * 0.5
            bullet['lifetime'] -= delta_time

            for pid, player in self.players.items():
                if pid != bullet['shooter_id']:
                    dist = math.sqrt(
                        (player['position']['x'] - bullet['position']['x'])**2 +
                        (player['position']['z'] - bullet['position']['z'])**2
                    )
                    if dist < 1:
                        expired.append(bullet_id)
                        player['scale_factor'] = max(player['scale_factor'] * 0.9, 0.1)
                        return {'type': 'player', 'id': pid}

            for nid, npc in self.npcs.items():
                dist = math.sqrt(
                    (npc.position['x'] - bullet['position']['x'])**2 +
                    (npc.position['z'] - bullet['position']['z'])**2
                )
                if dist < npc.size * npc.scale_factor:
                    expired.append(bullet_id)
                    npc.hit()
                    return {'type': 'npc', 'id': nid}

            if bullet['lifetime'] <= 0:
                expired.append(bullet_id)

        for bullet_id in expired:
            del self.bullets[bullet_id]
        return None

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
                if dx*dx + dz*dz > self.VIEW_DISTANCE*self.VIEW_DISTANCE:
                    continue
                chunk_key = f"{cx},{cz}"
                new_active_chunks.add(chunk_key)
                if chunk_key not in self.players[player_id].get('active_chunks', set()):
                    chunks[chunk_key] = self.chunk_generator.generate_chunk(cx, cz)

        self.players[player_id]['active_chunks'] = new_active_chunks
        return chunks

    def get_stats(self) -> Dict:
        return {
            "players_total": self.connections_total,
            "players_active": self.connections_active,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "chunks": self.chunk_generator.get_stats(),
            "time_of_day": self.time_of_day,
            "npcs_count": len(self.npcs),
            "bullets_count": len(self.bullets)
        }

class ClientConnection:
    def __init__(self, websocket: WebSocket, manager):
        self.websocket = websocket
        self.manager = manager
        self.player_id = f"player_{id(websocket)}"
        self.name = f"Player_{self.player_id[:8]}"
        self.last_activity = time.time()
        self.is_authenticated = False
        self.active = True
        self.sent_chunks = set()
        self.message_count = 0
        self.message_time = time.time()

    async def authenticate(self, timeout=5.0):
        try:
            data = await asyncio.wait_for(self.websocket.receive_text(), timeout=timeout)
            message = json.loads(data)
            if message['type'] == 'set_name' and 'name' in message.get('data', {}):
                self.name = message['data']['name']
                print(f"Authenticated player: {self.name} (ID: {self.player_id})")
                self.is_authenticated = True
                return True
        except asyncio.TimeoutError:
            print(f"Authentication timeout for {self.player_id}")
        except Exception as e:
            print(f"Authentication error for {self.player_id}: {e}")
        return False

    async def send_message(self, message: Dict):
        if not self.active:
            return False
        try:
            message_str = json.dumps(message)
            if len(message_str) > 1024:
                compressed = zlib.compress(message_str.encode(), 6)
                await self.websocket.send_bytes(compressed)
            else:
                await self.websocket.send_text(message_str)
            self.manager.game_state.messages_sent += 1
            return True
        except Exception as e:
            print(f"Error sending to {self.player_id}: {e}")
            self.active = False
            return False

    def check_rate_limit(self) -> bool:
        current_time = time.time()
        if current_time - self.message_time > 1.0:
            self.message_count = 1
            self.message_time = current_time
            return True
        self.message_count += 1
        if self.message_count > 20:
            print(f"Rate limit exceeded for {self.player_id}: {self.message_count} messages/second")
            return False
        return True

class ConnectionManager:
    def __init__(self, num_npcs: int = 10):
        self.connections: Dict[str, ClientConnection] = {}
        self.game_state = GameState(num_npcs=num_npcs)
        self.background_tasks = set()
        self.start_background_task(self.update_time_task())
        self.start_background_task(self.cleanup_inactive_players_task())
        self.start_background_task(self.keepalive_task())
        self.start_background_task(self.update_npcs_task())
        self.start_background_task(self.update_bullets_task())

    def start_background_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def connect(self, websocket: WebSocket) -> ClientConnection:
        await websocket.accept()
        client = ClientConnection(websocket, self)
        self.connections[client.player_id] = client
        print(f"New connection: {client.player_id}")

        await client.authenticate()
        self.game_state.add_player(client.player_id, client.name)

        chunks = self.game_state.get_nearby_chunks(client.player_id)

        players_data = {}
        for pid, player_data in self.game_state.players.items():
            if pid != client.player_id:
                players_data[pid] = {
                    'position': player_data['position'],
                    'rotation': player_data['rotation'],
                    'name': player_data['name']
                }

        await client.send_message({
            'type': 'init',
            'data': {
                'player_id': client.player_id,
                'chunks': chunks,
                'players': players_data,
                'npcs': self.game_state.get_npcs_data(),
                'time_of_day': self.game_state.time_of_day
            }
        })

        await self.broadcast({
            'type': 'player_joined',
            'data': {
                'player_id': client.player_id,
                'position': self.game_state.players[client.player_id]['position'],
                'rotation': self.game_state.players[client.player_id]['rotation'],
                'name': client.name
            }
        }, exclude=client.player_id)

        return client

    async def disconnect(self, client: ClientConnection):
        if client.player_id in self.connections:
            del self.connections[client.player_id]
        self.game_state.remove_player(client.player_id)
        await self.broadcast({
            'type': 'player_left',
            'data': {'player_id': client.player_id}
        })
        print(f"Disconnected: {client.player_id} ({client.name})")

    async def broadcast(self, message: Dict, exclude: Optional[str] = None):
        message_str = json.dumps(message)
        large_message = len(message_str) > 1024
        compressed = zlib.compress(message_str.encode(), 6) if large_message else None

        for player_id, client in list(self.connections.items()):
            if player_id != exclude and client.active:
                if large_message:
                    try:
                        await client.websocket.send_bytes(compressed)
                        self.game_state.messages_sent += 1
                    except Exception:
                        client.active = False
                else:
                    try:
                        await client.websocket.send_text(message_str)
                        self.game_state.messages_sent += 1
                    except Exception:
                        client.active = False

    async def update_player(self, client: ClientConnection, data: Dict):
        position = data.get('position')
        rotation = data.get('rotation', 0)
        self.game_state.update_player_position(client.player_id, position, rotation)

        current_chunks = self.game_state.get_nearby_chunks(client.player_id)
        new_chunks = {k: v for k, v in current_chunks.items() if k not in client.sent_chunks}
        client.sent_chunks.update(new_chunks.keys())

        if new_chunks:
            chunk_keys = list(new_chunks.keys())
            for i in range(0, len(chunk_keys), 3):
                batch = {k: new_chunks[k] for k in chunk_keys[i:i+3] if k in new_chunks}
                success = await client.send_message({
                    'type': 'chunks_update',
                    'data': {'chunks': batch}
                })
                if not success:
                    break
                if i + 3 < len(chunk_keys):
                    await asyncio.sleep(0.05)

        await self.broadcast({
            'type': 'position',
            'data': {
                'player_id': client.player_id,
                'position': position,
                'rotation': rotation,
                'name': client.name
            }
        }, exclude=client.player_id)

    async def handle_client(self, websocket: WebSocket):
        client = await self.connect(websocket)
        try:
            while client.active:
                try:
                    data = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                    client.last_activity = time.time()
                    self.game_state.messages_received += 1

                    message_data = None
                    if "text" in data:
                        message_data = json.loads(data["text"])
                    elif "bytes" in data:
                        decompressed = zlib.decompress(data["bytes"])
                        message_data = json.loads(decompressed.decode())

                    if not message_data:
                        continue

                    if not client.check_rate_limit():
                        await client.send_message({
                            'type': 'warning',
                            'data': {'message': 'Rate limit exceeded. Please slow down requests.'}
                        })
                        continue

                    if message_data['type'] == 'position':
                        await self.update_player(client, message_data['data'])
                    elif message_data['type'] == 'ping':
                        await client.send_message({
                            'type': 'pong',
                            'data': {'server_time': time.time()}
                        })
                    elif message_data['type'] == 'shoot':
                        bullet_id = f"bullet_{client.player_id}_{time.time()}"
                        self.game_state.add_bullet(
                            bullet_id,
                            message_data['data']['position'],
                            message_data['data']['direction'],
                            client.player_id
                        )
                    elif message_data['type'] == 'hit':
                        await self.broadcast({
                            'type': 'hit',
                            'data': {
                                'target_id': message_data['data']['target_id'],
                                'target_type': message_data['data']['target_type']
                            }
                        })

                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    print(f"WebSocket disconnected: {client.player_id}")
                    break
                except Exception as e:
                    print(f"Error handling client {client.player_id}: {e}")
                    if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                        break
        finally:
            await self.disconnect(client)

    async def update_time_task(self):
        while True:
            try:
                self.game_state.time_of_day = (self.game_state.time_of_day + 1 / (self.game_state.DAY_NIGHT_CYCLE * 10)) % 1
                if int(self.game_state.time_of_day * self.game_state.DAY_NIGHT_CYCLE) % 10 == 0:
                    await self.broadcast({
                        'type': 'time_update',
                        'data': {'time_of_day': self.game_state.time_of_day}
                    })
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in update_time_task: {e}")
                await asyncio.sleep(1)

    async def update_npcs_task(self):
        while True:
            try:
                self.game_state.update_npcs()
                if self.connections:
                    await self.broadcast({
                        'type': 'npcs_update',
                        'data': {'npcs': self.game_state.get_npcs_data()}
                    })
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"Error in update_npcs_task: {e}")
                await asyncio.sleep(1)

    async def update_bullets_task(self):
        while True:
            try:
                delta_time = 0.1
                hit = self.game_state.update_bullets(delta_time)
                if hit:
                    await self.broadcast({
                        'type': 'hit',
                        'data': {
                            'target_id': hit['id'],
                            'target_type': hit['type']
                        }
                    })
                await asyncio.sleep(delta_time)
            except Exception as e:
                print(f"Error in update_bullets_task: {e}")
                await asyncio.sleep(1)

    async def cleanup_inactive_players_task(self):
        while True:
            try:
                current_time = time.time()
                inactive_clients = []
                for player_id, client in list(self.connections.items()):
                    if current_time - client.last_activity > 120:
                        inactive_clients.append(client)
                for client in inactive_clients:
                    print(f"Removing inactive client: {client.player_id} ({client.name})")
                    await self.disconnect(client)
                await asyncio.sleep(30)
            except Exception as e:
                print(f"Error in cleanup_inactive_players_task: {e}")
                await asyncio.sleep(30)

    async def keepalive_task(self):
        while True:
            try:
                current_time = time.time()
                for player_id, client in list(self.connections.items()):
                    if current_time - client.last_activity > 10 and client.active:
                        try:
                            await client.send_message({
                                'type': 'ping',
                                'data': {'server_time': current_time}
                            })
                        except Exception as e:
                            print(f"Error sending keepalive to {player_id}: {e}")
                            client.active = False
                await asyncio.sleep(15)
            except Exception as e:
                print(f"Error in keepalive_task: {e}")
                await asyncio.sleep(15)

manager = ConnectionManager(num_npcs=15)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.handle_client(websocket)

@app.get("/server-stats")
async def get_server_stats():
    return manager.game_state.get_stats()

@app.get("/set-npcs/{count}")
async def set_npcs_count(count: int):
    if count < 0 or count > 100:
        return {"error": "NPC count must be between 0 and 100"}
    
    manager.game_state.npcs.clear()
    manager.game_state.init_npcs(count)
    
    return {"message": f"NPC count set to {count}", "npcs": len(manager.game_state.npcs)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)