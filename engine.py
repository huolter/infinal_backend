from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Dict, List, Tuple, Set
import asyncio
import random
import math
import noise
import os
import time
import zlib
from starlette.websockets import WebSocketDisconnect
from collections import OrderedDict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gualterio.com", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LRUCache(OrderedDict):
    """Simple LRU cache implementation"""
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
        self.chunks = LRUCache(maxsize=2000)  # Store up to 2000 chunks in memory
        self.DENSITY = 0.1
        self.seed = random.randint(0, 1000000)
        self.biome_scale = 100.0
        self.terrain_scale = 50.0
        
        # Track statistics for debugging
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
        
        # Check cache first
        if chunk_key in self.chunks:
            self.cache_hits += 1
            return self.chunks[chunk_key]
            
        self.generated_chunks += 1
        
        # Generate new chunk
        biome = self.get_biome(chunk_x, chunk_z)
        chunk = {'terrain': [], 'features': [], 'entities': [], 'biome': biome}

        # Generate terrain and features
        for z in range(16):
            terrain_row = []
            feature_row = []
            for x in range(16):
                wx = x + chunk_x * 16
                wz = z + chunk_z * 16
                height = self.get_noise(wx, wz, self.terrain_scale) * 10
                terrain_row.append(height)
                
                # Only add features with certain probability
                if random.random() < self.DENSITY:
                    feature_type = self.get_feature_type(wx, wz, biome)
                    feature_row.append(feature_type)
                else:
                    feature_row.append(0)
            chunk['terrain'].append(terrain_row)
            chunk['features'].append(feature_row)

        # Generate entities and water features
        chunk['entities'] = self.generate_entities(chunk_x, chunk_z, biome)
        
        # Add water to plains biomes occasionally
        if biome == 'plains' and random.random() < 0.1:
            water_size = random.randint(3, 6)
            water_x = random.randint(0, 15 - water_size)
            water_z = random.randint(0, 15 - water_size)
            chunk['water'] = {'size': water_size, 'position': {'x': water_x, 'z': water_z}}

        # Cache and return the chunk
        self.chunks[chunk_key] = chunk
        return chunk

    def get_feature_type(self, x: float, z: float, biome: str) -> Dict:
        # Use consistent random seed based on position for deterministic features
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
        # Only add entities with a certain probability
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
            
    def get_stats(self) -> Dict:
        return {
            "total_chunks_generated": self.generated_chunks,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.chunks),
        }

class GameState:
    def __init__(self):
        self.players: Dict[str, Dict] = {}
        self.chunk_generator = ChunkGenerator()
        self.VIEW_DISTANCE = 3  # Increased view distance for better experience
        self.time_of_day = 0
        self.DAY_NIGHT_CYCLE = 600  # Longer day-night cycle (10 minutes)
        self.last_activity = {}  # Track last activity for each player
        
        # Track statistics
        self.connections_total = 0
        self.connections_active = 0
        self.messages_received = 0
        self.messages_sent = 0

    def add_player(self, player_id: str, name: str = "Unnamed", position: Dict = None):
        if position is None:
            position = {'x': 0, 'y': 1.7, 'z': 0}
        
        self.players[player_id] = {
            'position': position,
            'rotation': 0,
            'active_chunks': set(),
            'name': name,
            'joined_at': time.time()
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

    def get_nearby_chunks(self, player_id: str) -> Dict[str, Dict]:
        if player_id not in self.players:
            return {}
            
        pos = self.players[player_id]['position']
        chunk_x = math.floor(pos['x'] / 16)
        chunk_z = math.floor(pos['z'] / 16)
        
        chunks = {}
        new_active_chunks = set()
        
        # Generate all chunks within VIEW_DISTANCE
        for dx in range(-self.VIEW_DISTANCE, self.VIEW_DISTANCE + 1):
            for dz in range(-self.VIEW_DISTANCE, self.VIEW_DISTANCE + 1):
                # Calculate chunk coordinates
                cx = chunk_x + dx
                cz = chunk_z + dz
                
                # Skip chunks that are too far (use circular distance)
                if dx*dx + dz*dz > self.VIEW_DISTANCE*self.VIEW_DISTANCE:
                    continue
                
                chunk_key = f"{cx},{cz}"
                new_active_chunks.add(chunk_key)
                
                # Only generate chunks that weren't already active
                if chunk_key not in self.players[player_id]['active_chunks']:
                    chunks[chunk_key] = self.chunk_generator.generate_chunk(cx, cz)
        
        # Update player's active chunks
        self.players[player_id]['active_chunks'] = new_active_chunks
        return chunks
        
    def get_stats(self) -> Dict:
        return {
            "players_total": self.connections_total,
            "players_active": self.connections_active,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "chunks": self.chunk_generator.get_stats(),
            "time_of_day": self.time_of_day
        }

    async def update_time(self, manager):
        while True:
            try:
                # Update time of day more slowly
                self.time_of_day = (self.time_of_day + 1 / (self.DAY_NIGHT_CYCLE * 10)) % 1
                
                # Only broadcast time updates every second
                if int(self.time_of_day * self.DAY_NIGHT_CYCLE) % 10 == 0:
                    await manager.broadcast({
                        'type': 'time_update',
                        'data': {'time_of_day': self.time_of_day}
                    })
                    
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in update_time: {e}")
                await asyncio.sleep(1)  # Back off on error
                
    async def cleanup_inactive_players(self, manager):
        """Remove players who haven't sent updates in a while"""
        while True:
            try:
                current_time = time.time()
                inactive_players = []
                
                for player_id, last_time in self.last_activity.items():
                    # If inactive for more than 2 minutes
                    if current_time - last_time > 120:
                        inactive_players.append(player_id)
                
                for player_id in inactive_players:
                    print(f"Removing inactive player: {player_id}")
                    self.remove_player(player_id)
                    
                    # Broadcast player left message
                    await manager.broadcast({
                        'type': 'player_left',
                        'data': {'player_id': player_id, 'reason': 'inactive'}
                    })
                    
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in cleanup_inactive_players: {e}")
                await asyncio.sleep(30)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}  # player_id -> websocket
        self.player_ids: Dict[WebSocket, str] = {}  # websocket -> player_id
        self.last_message_time: Dict[str, float] = {}  # player_id -> timestamp
        self.game_state = GameState()
        
        # Start background tasks
        asyncio.create_task(self.game_state.update_time(self))
        asyncio.create_task(self.game_state.cleanup_inactive_players(self))
        asyncio.create_task(self.keepalive())
        
        # Compression level (0-9, higher = better compression but slower)
        self.compression_level = 6
        
        # Rate limiting
        self.rate_limits = {}  # player_id -> {count, first_request}
        self.MAX_MESSAGES_PER_SECOND = 20

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        
        # Generate a unique player ID
        player_id = str(id(websocket))
        self.active_connections[player_id] = websocket
        self.player_ids[websocket] = player_id
        self.last_message_time[player_id] = time.time()
        
        # Wait for player name with timeout
        try:
            name = await self.get_player_name(websocket)
        except (WebSocketDisconnect, asyncio.TimeoutError, json.JSONDecodeError) as e:
            name = f"Player_{player_id[:8]}"
            print(f"Client connection issue: {e}, assigned name: {name}")
        
        # Add player to game state
        self.game_state.add_player(player_id, name)
        
        # Get initial chunks for new player
        chunks = self.game_state.get_nearby_chunks(player_id)
        
        # Prepare player data for sending
        players_data = {}
        for pid, player_data in self.game_state.players.items():
            if pid != player_id:  # Don't include the new player
                player_data_copy = {
                    'position': player_data['position'],
                    'rotation': player_data['rotation'],
                    'name': player_data['name']
                }
                players_data[pid] = player_data_copy
        
        try:
            # Send initial game state to new player
            await self.send_to_client(websocket, {
                'type': 'init',
                'data': {
                    'player_id': player_id,
                    'chunks': chunks,
                    'players': players_data,
                    'time_of_day': self.game_state.time_of_day
                }
            })
            
            # Notify other players about the new player
            await self.broadcast({
                'type': 'player_joined',
                'data': {
                    'player_id': player_id,
                    'position': self.game_state.players[player_id]['position'],
                    'rotation': self.game_state.players[player_id]['rotation'],
                    'name': self.game_state.players[player_id]['name']
                }
            }, exclude_player=player_id)
            
            print(f"Player {player_id} ({name}) connected")
            
        except WebSocketDisconnect:
            print(f"Client {player_id} disconnected during init")
            await self.disconnect(websocket)

    async def get_player_name(self, websocket: WebSocket) -> str:
        """Get player name from initial message with timeout"""
        try:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            message = json.loads(data)
            if message['type'] == 'set_name' and 'name' in message.get('data', {}):
                return message['data']['name']
        except Exception as e:
            print(f"Error getting player name: {e}")
        
        # Default name if no valid name message received
        return f"Player_{id(websocket)}"

    async def disconnect(self, websocket: WebSocket):
        """Handle client disconnection"""
        if websocket in self.player_ids:
            player_id = self.player_ids[websocket]
            
            # Clean up connection tracking
            if player_id in self.active_connections:
                del self.active_connections[player_id]
            del self.player_ids[websocket]
            if player_id in self.last_message_time:
                del self.last_message_time[player_id]
            if player_id in self.rate_limits:
                del self.rate_limits[player_id]
            
            # Remove from game state
            self.game_state.remove_player(player_id)
            
            # Notify other clients
            try:
                await self.broadcast({
                    'type': 'player_left',
                    'data': {'player_id': player_id}
                })
                print(f"Player {player_id} disconnected")
            except Exception as e:
                print(f"Error broadcasting player_left: {e}")

    async def send_to_client(self, websocket: WebSocket, message: Dict):
        """Send a message to a specific client with compression"""
        try:
            # Compress message if it's large
            message_str = json.dumps(message)
            
            if len(message_str) > 1024:  # Only compress messages larger than 1KB
                compressed = zlib.compress(message_str.encode(), self.compression_level)
                await websocket.send_bytes(compressed)
            else:
                await websocket.send_text(message_str)
                
            self.game_state.messages_sent += 1
            
        except Exception as e:
            print(f"Error sending to client: {e}")
            # Let the caller handle any disconnect exceptions
            raise

    async def broadcast(self, message: Dict, exclude_player: str = None):
        """Broadcast a message to all clients except the excluded one"""
        message_str = json.dumps(message)
        large_message = len(message_str) > 1024
        
        if large_message:
            # Pre-compress large messages once
            compressed = zlib.compress(message_str.encode(), self.compression_level)
        
        # Send to each client in parallel
        tasks = []
        for player_id, websocket in self.active_connections.items():
            if player_id != exclude_player:
                if large_message:
                    tasks.append(self.send_bytes_to_client(websocket, compressed))
                else:
                    tasks.append(self.send_text_to_client(websocket, message_str))
        
        if tasks:
            # Wait for all sends to complete, ignoring individual errors
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful messages
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            self.game_state.messages_sent += success_count
            
            # Handle disconnected clients
            for i, result in enumerate(results):
                if isinstance(result, (WebSocketDisconnect, ConnectionRefusedError)):
                    # The websocket is already gone, find which one it was
                    for ws in list(self.player_ids.keys()):
                        try:
                            await ws.receive_text()
                        except:
                            await self.disconnect(ws)
                            break

    async def send_text_to_client(self, websocket: WebSocket, message_str: str):
        """Helper to send text message with error handling"""
        try:
            await websocket.send_text(message_str)
            return True
        except Exception as e:
            if websocket in self.player_ids:
                await self.disconnect(websocket)
            return e

    async def send_bytes_to_client(self, websocket: WebSocket, compressed_bytes: bytes):
        """Helper to send binary message with error handling"""
        try:
            await websocket.send_bytes(compressed_bytes)
            return True
        except Exception as e:
            if websocket in self.player_ids:
                await self.disconnect(websocket)
            return e

    async def update_player(self, websocket: WebSocket, player_id: str, data: Dict):
        """Update player position and send nearby chunks"""
        # Update player state
        position = data.get('position')
        rotation = data.get('rotation', 0)
        self.game_state.update_player_position(player_id, position, rotation)
        
        # Check which chunks the player needs
        current_chunks = self.game_state.get_nearby_chunks(player_id)
        
        # Only send new chunks that weren't previously active
        new_chunks = {}
        for key, chunk in current_chunks.items():
            if key not in self.game_state.players[player_id].get('sent_chunks', set()):
                new_chunks[key] = chunk
        
        # Update sent chunks tracking
        if not hasattr(self.game_state.players[player_id], 'sent_chunks'):
            self.game_state.players[player_id]['sent_chunks'] = set()
        self.game_state.players[player_id]['sent_chunks'].update(new_chunks.keys())
        
        # Send chunks in small batches if there are many
        try:
            if new_chunks:
                chunk_keys = list(new_chunks.keys())
                for i in range(0, len(chunk_keys), 5):  # Send 5 chunks at a time
                    batch = {k: new_chunks[k] for k in chunk_keys[i:i+5] if k in new_chunks}
                    await self.send_to_client(websocket, {
                        'type': 'chunks_update',
                        'data': {'chunks': batch}
                    })
                    if i + 5 < len(chunk_keys):
                        await asyncio.sleep(0.05)  # Small delay between batches
            
            # Broadcast position update to other players
            await self.broadcast({
                'type': 'position',
                'data': {
                    'player_id': player_id,
                    'position': position,
                    'rotation': rotation,
                    'name': self.game_state.players[player_id]['name']
                }
            }, exclude_player=player_id)
            
        except WebSocketDisconnect:
            print(f"Client {player_id} disconnected during update_player")
            await self.disconnect(websocket)

    async def keepalive(self):
        """Send periodic keepalive messages to maintain connections"""
        while True:
            await asyncio.sleep(15)  # Check every 15 seconds
            try:
                current_time = time.time()
                
                # Only ping clients that haven't sent a message recently
                for player_id, last_time in self.last_message_time.items():
                    if current_time - last_time > 10 and player_id in self.active_connections:
                        try:
                            # Send a ping to keep the connection alive
                            await self.send_to_client(self.active_connections[player_id], {
                                'type': 'ping',
                                'data': {'server_time': current_time}
                            })
                        except Exception as e:
                            print(f"Error in keepalive for player {player_id}: {e}")
                            # The disconnect will be handled by the send_to_client exception
            except Exception as e:
                print(f"Error in keepalive: {e}")
                
    def check_rate_limit(self, player_id: str) -> bool:
        """Check if a player is sending too many messages"""
        current_time = time.time()
        
        if player_id not in self.rate_limits:
            self.rate_limits[player_id] = {"count": 1, "first_request": current_time}
            return True
            
        rate_data = self.rate_limits[player_id]
        
        # Reset counter if more than 1 second has passed
        if current_time - rate_data["first_request"] > 1.0:
            rate_data["count"] = 1
            rate_data["first_request"] = current_time
            return True
            
        # Increment counter
        rate_data["count"] += 1
        
        # Check if limit exceeded
        if rate_data["count"] > self.MAX_MESSAGES_PER_SECOND:
            print(f"Rate limit exceeded for player {player_id}: {rate_data['count']} messages/second")
            return False
            
        return True

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for a message from the client
            try:
                data = await websocket.receive()
                
                # Handle both text and binary messages
                if "text" in data:
                    message_data = json.loads(data["text"])
                elif "bytes" in data:
                    # Decompress binary messages
                    decompressed = zlib.decompress(data["bytes"])
                    message_data = json.loads(decompressed.decode())
                else:
                    continue
                
                # Get player ID from the connection
                if websocket not in manager.player_ids:
                    print("Received message from unregistered websocket")
                    continue
                    
                player_id = manager.player_ids[websocket]
                
                # Update last message time
                manager.last_message_time[player_id] = time.time()
                manager.game_state.messages_received += 1
                
                # Check rate limiting
                if not manager.check_rate_limit(player_id):
                    # Send warning message
                    await manager.send_to_client(websocket, {
                        'type': 'warning',
                        'data': {'message': 'Rate limit exceeded. Please slow down requests.'}
                    })
                    continue
                
                # Process messages based on type
                if message_data['type'] == 'position':
                    await manager.update_player(websocket, player_id, message_data['data'])
                elif message_data['type'] == 'ping':
                    # Respond to client pings
                    await manager.send_to_client(websocket, {
                        'type': 'pong',
                        'data': {'server_time': time.time()}
                    })
                
            except WebSocketDisconnect:
                print(f"WebSocket disconnected for player {manager.player_ids.get(websocket, 'unknown')}")
                await manager.disconnect(websocket)
                break
                
            except json.JSONDecodeError:
                print("Error decoding JSON message")
                continue
                
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
                
    except Exception as e:
        print(f"Websocket error: {e}")
        await manager.disconnect(websocket)

@app.get("/server-stats")
async def get_server_stats():
    """Endpoint to get server statistics"""
    return manager.game_state.get_stats()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
