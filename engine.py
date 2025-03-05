from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import Dict, List, Tuple, Set, Optional
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

class Entity:
    """Base class for all dynamic entities in the game"""
    def __init__(self, entity_id, entity_type, position, properties=None):
        self.id = entity_id
        self.type = entity_type
        self.position = position
        self.properties = properties or {}
        self.velocity = {'x': 0, 'y': 0, 'z': 0}
        self.rotation = 0
        self.created_at = time.time()
        self.last_update = self.created_at
        self.active = True
        
    def update(self, dt):
        """Update entity position based on velocity"""
        self.position['x'] += self.velocity['x'] * dt
        self.position['y'] += self.velocity['y'] * dt
        self.position['z'] += self.velocity['z'] * dt
        self.last_update = time.time()
        
    def to_dict(self):
        """Convert entity to dictionary for sending to clients"""
        return {
            'id': self.id,
            'type': self.type,
            'position': self.position,
            'velocity': self.velocity,
            'rotation': self.rotation,
            'properties': self.properties
        }

class RunningEntity(Entity):
    """Entity that runs/walks across the landscape"""
    def __init__(self, entity_id, position):
        # Randomly select a running entity type
        entity_type = random.choice([
            'fox', 'deer', 'rabbit', 'wolf', 'goat', 
            'camel', 'scorpion', 'snake', 'eagle'
        ])
        
        # Generate random properties
        size = random.uniform(0.5, 2.0)
        color = f"#{random.randint(0, 0xFFFFFF):06x}"  # Random color
        
        properties = {
            'size': size,
            'color': color,
            'speed': random.uniform(0.5, 3.0),
            'bounceHeight': random.uniform(0, 0.3),
            'legCount': random.randint(2, 8),
            'tailLength': random.uniform(0, 1.0)
        }
        
        super().__init__(entity_id, entity_type, position, properties)
        
        # Set random velocity (only in x-z plane for runners)
        angle = random.uniform(0, math.pi * 2)
        speed = properties['speed']
        self.velocity = {
            'x': math.cos(angle) * speed,
            'y': 0,
            'z': math.sin(angle) * speed
        }
        self.rotation = angle
        
        # Direction change parameters
        self.direction_change_interval = random.uniform(3, 10)
        self.next_direction_change = time.time() + self.direction_change_interval
        
    def update(self, dt):
        """Update running entity with random direction changes"""
        current_time = time.time()
        
        # Change direction occasionally
        if current_time > self.next_direction_change:
            # Get current speed
            current_speed = math.sqrt(self.velocity['x']**2 + self.velocity['z']**2)
            
            # Set new random direction
            angle = random.uniform(0, math.pi * 2)
            self.velocity['x'] = math.cos(angle) * current_speed
            self.velocity['z'] = math.sin(angle) * current_speed
            self.rotation = angle
            
            # Schedule next direction change
            self.direction_change_interval = random.uniform(3, 10)
            self.next_direction_change = current_time + self.direction_change_interval
            
        # Add bouncing effect for running
        bounce_cycle = math.sin(current_time * 5 * self.properties['speed'])
        self.position['y'] = max(0, self.properties['bounceHeight'] * bounce_cycle)
        
        # Call parent update to apply velocity
        super().update(dt)

class FlyingEntity(Entity):
    """Entity that flies through the air with geometric shapes"""
    def __init__(self, entity_id, position):
        # Select a random geometric shape
        entity_type = random.choice([
            'cube', 'sphere', 'tetrahedron', 'octahedron', 
            'cylinder', 'torus', 'pyramid', 'cone'
        ])
        
        # Generate random properties
        size = random.uniform(1.0, 5.0)
        primary_color = f"#{random.randint(0, 0xFFFFFF):06x}"
        secondary_color = f"#{random.randint(0, 0xFFFFFF):06x}"
        emission_intensity = random.uniform(0.5, 2.0)
        
        properties = {
            'size': size,
            'primaryColor': primary_color,
            'secondaryColor': secondary_color,
            'emissive': random.choice([True, False]),
            'emissionIntensity': emission_intensity,
            'rotationSpeed': random.uniform(-2, 2),
            'particleTrail': random.choice([True, False]),
            'particleColor': f"#{random.randint(0, 0xFFFFFF):06x}"
        }
        
        # Flying entities start higher
        position['y'] = random.uniform(10, 50)
        
        super().__init__(entity_id, entity_type, position, properties)
        
        # Set random 3D velocity for flyers
        angle_horizontal = random.uniform(0, math.pi * 2)
        angle_vertical = random.uniform(-math.pi / 6, math.pi / 6)  # Mostly horizontal
        speed = random.uniform(2.0, 10.0)
        
        self.velocity = {
            'x': math.cos(angle_horizontal) * math.cos(angle_vertical) * speed,
            'y': math.sin(angle_vertical) * speed,
            'z': math.sin(angle_horizontal) * math.cos(angle_vertical) * speed
        }
        
        # Change behavior occasionally
        self.behavior_change_interval = random.uniform(5, 15)
        self.next_behavior_change = time.time() + self.behavior_change_interval
        
    def update(self, dt):
        """Update flying entity with more complex behavior"""
        current_time = time.time()
        
        # Change behavior occasionally
        if current_time > self.next_behavior_change:
            # Random behavior changes
            behavior = random.choice(['soar', 'dive', 'zigzag', 'hover'])
            
            if behavior == 'soar':
                # Soar upward
                self.velocity['y'] = random.uniform(2, 5)
                self.velocity['x'] *= 0.8
                self.velocity['z'] *= 0.8
            elif behavior == 'dive':
                # Dive downward
                self.velocity['y'] = -random.uniform(3, 7)
                self.velocity['x'] *= 1.2
                self.velocity['z'] *= 1.2
            elif behavior == 'zigzag':
                # Sharp direction change
                angle = random.uniform(0, math.pi * 2)
                speed = math.sqrt(self.velocity['x']**2 + self.velocity['z']**2)
                self.velocity['x'] = math.cos(angle) * speed * 1.5
                self.velocity['z'] = math.sin(angle) * speed * 1.5
            elif behavior == 'hover':
                # Slow down and hover
                self.velocity['x'] *= 0.2
                self.velocity['y'] *= 0.2
                self.velocity['z'] *= 0.2
            
            # Schedule next behavior change
            self.behavior_change_interval = random.uniform(5, 15)
            self.next_behavior_change = current_time + self.behavior_change_interval
        
        # Keep flying entities above ground
        if self.position['y'] < 5 and self.velocity['y'] < 0:
            self.velocity['y'] *= -0.8  # Bounce up if getting too low
        
        # Cap maximum height
        if self.position['y'] > 100:
            self.velocity['y'] = -abs(self.velocity['y'])
            
        # Update rotation
        self.rotation += self.properties['rotationSpeed'] * dt
        
        # Call parent update to apply velocity
        super().update(dt)

class AbstractEntity(Entity):
    """Abstract, crazy entities with unusual properties and behaviors"""
    def __init__(self, entity_id, position):
        # Create bizarre entity types
        entity_type = random.choice([
            'vortex', 'hyperblob', 'quantumFlux', 'dimensionTear',
            'cosmicJellyfish', 'realityGlitch', 'thoughtForm', 'dreamFragment'
        ])
        
        # Generate wild properties
        num_colors = random.randint(1, 5)
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_colors)]
        
        properties = {
            'colors': colors,
            'mainSize': random.uniform(1.0, 8.0),
            'segments': random.randint(3, 20),
            'pulseRate': random.uniform(0.5, 5.0),
            'twistFactor': random.uniform(-3, 3),
            'morphCycle': random.uniform(2, 10),
            'spikiness': random.uniform(0, 1),
            'transparency': random.uniform(0.2, 0.9),
            'soundEmission': random.choice([True, False]),
            'soundFrequency': random.uniform(50, 2000) if random.random() > 0.5 else None,
            'teleportation': random.random() > 0.8,  # 20% chance to teleport
            'warpReality': random.random() > 0.9,    # 10% chance to warp reality
            'tentacleCount': random.randint(0, 12),
            'tentacleLength': random.uniform(0, 5)
        }
        
        # Random starting position
        position['y'] = random.uniform(0, 30)
        
        super().__init__(entity_id, entity_type, position, properties)
        
        # Use complex motion patterns
        self.velocity = {
            'x': random.uniform(-2, 2),
            'y': random.uniform(-2, 2),
            'z': random.uniform(-2, 2)
        }
        
        # State for special behaviors
        self.phase = random.uniform(0, math.pi * 2)  # For cyclical behaviors
        self.state_change_time = time.time() + random.uniform(2, 8)
        
    def update(self, dt):
        """Update abstract entity with bizarre, unpredictable behavior"""
        current_time = time.time()
        self.phase += dt * self.properties['pulseRate']
        
        # Apply weird movement patterns
        if self.properties['teleportation'] and random.random() < 0.01:
            # Occasional teleportation
            self.position['x'] += random.uniform(-20, 20)
            self.position['y'] += random.uniform(-10, 10)
            self.position['z'] += random.uniform(-20, 20)
            
            # Keep above ground
            self.position['y'] = max(2, self.position['y'])
        
        # Oscillating size
        current_size = self.properties['mainSize'] * (0.8 + 0.4 * math.sin(self.phase))
        self.properties['currentSize'] = current_size
        
        # Apply sinusoidal motion
        if random.random() < 0.05:
            # Occasionally change velocity
            self.velocity = {
                'x': random.uniform(-3, 3),
                'y': random.uniform(-2, 2),
                'z': random.uniform(-3, 3)
            }
        
        # Add sine wave movement
        self.position['x'] += math.sin(self.phase * 0.3) * 0.1
        self.position['y'] += math.sin(self.phase * 0.5) * 0.1
        self.position['z'] += math.sin(self.phase * 0.7) * 0.1
        
        # Keep above ground with a minimum bounce
        if self.position['y'] < 1:
            self.position['y'] = 1
            self.velocity['y'] = abs(self.velocity['y']) + random.uniform(0.5, 2)
        
        # Call parent update to apply velocity
        super().update(dt)

class ChunkGenerator:
    def __init__(self, entity_manager):
        self.chunks = LRUCache(maxsize=2000)  # Store up to 2000 chunks in memory
        self.DENSITY = 0.1
        self.seed = random.randint(0, 1000000)
        self.biome_scale = 100.0
        self.terrain_scale = 50.0
        self.entity_manager = entity_manager
        
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

        # Generate static entities and water features
        chunk['static_entities'] = self.generate_static_entities(chunk_x, chunk_z, biome)
        
        # Add water to plains biomes occasionally
        if biome == 'plains' and random.random() < 0.1:
            water_size = random.randint(3, 6)
            water_x = random.randint(0, 15 - water_size)
            water_z = random.randint(0, 15 - water_size)
            chunk['water'] = {'size': water_size, 'position': {'x': water_x, 'z': water_z}}

        # Add dynamic entities occasionally
        if random.random() < 0.3:  # 30% chance per chunk
            # Create 1-3 dynamic entities in this chunk
            for _ in range(random.randint(1, 3)):
                # Random position within the chunk
                entity_x = (chunk_x * 16) + random.uniform(0, 16) 
                entity_z = (chunk_z * 16) + random.uniform(0, 16)
                
                # Create a dynamic entity
                self.entity_manager.create_random_entity({
                    'x': entity_x,
                    'y': 0,  # Will be adjusted by the entity type
                    'z': entity_z
                })

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

    def generate_static_entities(self, chunk_x: int, chunk_z: int, biome: str) -> List[Dict]:
        entities = []
        # Only add entities with a certain probability
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
            "cache_size": len(self.chunks),
        }

class EntityManager:
    """Manages all dynamic entities in the game"""
    def __init__(self):
        self.entities = {}  # id -> entity
        self.entity_counter = 0
        self.last_update = time.time()
        self.max_entities = 200  # Maximum number of entities to avoid overload
        
    def create_entity(self, entity_type, position):
        """Create a new entity of the specified type"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        
        if entity_type == "runner":
            entity = RunningEntity(entity_id, position)
        elif entity_type == "flyer":
            entity = FlyingEntity(entity_id, position)
        elif entity_type == "abstract":
            entity = AbstractEntity(entity_id, position)
        else:
            return None
            
        self.entities[entity_id] = entity
        return entity
        
    def create_random_entity(self, position):
        """Create a random entity at the specified position"""
        # Enforce entity limit
        if len(self.entities) >= self.max_entities:
            # Remove oldest entity
            oldest_id = min(self.entities.items(), key=lambda x: x[1].created_at)[0]
            del self.entities[oldest_id]
            
        # Choose a random entity type with weighted probabilities
        entity_type = random.choices(
            ["runner", "flyer", "abstract"],
            weights=[0.5, 0.3, 0.2],  # 50% runners, 30% flyers, 20% abstract
            k=1
        )[0]
        
        return self.create_entity(entity_type, position)
        
    def update(self):
        """Update all entities"""
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Cap dt to avoid large jumps
        dt = min(dt, 0.1)
        
        entities_to_remove = []
        for entity_id, entity in self.entities.items():
            try:
                entity.update(dt)
                
                # Check if entity is too far from origin (limit world size)
                distance = math.sqrt(
                    entity.position['x']**2 + 
                    entity.position['z']**2
                )
                
                if distance > 1000:  # 1000 units from origin
                    entities_to_remove.append(entity_id)
            except Exception as e:
                print(f"Error updating entity {entity_id}: {e}")
                entities_to_remove.append(entity_id)
                
        # Remove entities that are out of bounds
        for entity_id in entities_to_remove:
            del self.entities[entity_id]
            
    def get_entities_in_range(self, position, radius):
        """Get all entities within a certain radius of a position"""
        entities_in_range = []
        
        for entity in self.entities.values():
            # Calculate distance (only in x-z plane to match chunk loading)
            dx = entity.position['x'] - position['x']
            dz = entity.position['z'] - position['z']
            distance = math.sqrt(dx*dx + dz*dz)
            
            if distance <= radius:
                entities_in_range.append(entity.to_dict())
                
        return entities_in_range
        
    def get_stats(self):
        """Get statistics about entities"""
        entity_types = {}
        for entity in self.entities.values():
            if entity.type in entity_types:
                entity_types[entity.type] += 1
            else:
                entity_types[entity.type] = 1
                
        return {
            "total_entities": len(self.entities),
            "entity_types": entity_types
        }

class GameState:
    def __init__(self):
        self.entity_manager = EntityManager()
        self.players: Dict[str, Dict] = {}
        self.chunk_generator = ChunkGenerator(self.entity_manager)
        self.VIEW_DISTANCE = 3  # Increased view distance for better experience
        self.time_of_day = 0
        self.DAY_NIGHT_CYCLE = 600  # Longer day-night cycle (10 minutes)
        self.last_activity = {}  # Track last activity for each player
        
        # Track statistics
        self.connections_total = 0
        self.connections_active = 0
        self.messages_received = 0
        self.messages_sent = 0
        
        # Last update time for entity simulation
        self.last_entity_update = time.time()

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
                if chunk_key not in self.players[player_id].get('active_chunks', set()):
                    chunks[chunk_key] = self.chunk_generator.generate_chunk(cx, cz)
        
        # Update player's active chunks
        self.players[player_id]['active_chunks'] = new_active_chunks
        return chunks
    
    def get_nearby_entities(self, player_id: str) -> List[Dict]:
        """Get all dynamic entities near a player"""
        if player_id not in self.players:
            return []
            
        pos = self.players[player_id]['position']
        
        # Match VIEW_DISTANCE but in world units
        radius = self.VIEW_DISTANCE * 16
        
        # Get entities within range
        return self.entity_manager.get_entities_in_range(pos, radius)
    
    def update_entities(self):
        """Update all dynamic entities"""
        current_time = time.time()
        if current_time - self.last_entity_update >= 0.05:  # 20 updates per second
            self.entity_manager.update()
            self.last_entity_update = current_time
        
    def get_stats(self) -> Dict:
        return {
            "players_total": self.connections_total,
            "players_active": self.connections_active,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "chunks": self.chunk_generator.get_stats(),
            "entities": self.entity_manager.get_stats(),
            "time_of_day": self.time_of_day
        }

class ClientConnection:
    """Manages a single client connection"""
    def __init__(self, websocket: WebSocket, manager):
        self.websocket = websocket
        self.manager = manager
        self.player_id = f"player_{id(websocket)}"
        self.name = f"Player_{self.player_id[:8]}"
        self.last_activity = time.time()
        self.is_authenticated = False
        self.active = True
        self.sent_chunks = set()
        self.known_entities = set()  # Track entities the client knows about
        self.message_count = 0
        self.message_time = time.time()
        
    async def authenticate(self, timeout=5.0):
        """Try to get player name from authentication message"""
        try:
            # Wait for name with timeout
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
        """Send a message to this client with error handling"""
        if not self.active:
            return False
            
        try:
            # Compress large messages
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
        """Check if client is sending too many messages"""
        current_time = time.time()
        if current_time - self.message_time > 1.0:
            # Reset counter if more than 1 second has passed
            self.message_count = 1
            self.message_time = current_time
            return True
            
        # Increment counter
        self.message_count += 1
        
        # Check if limit exceeded (20 msgs/second)
        if self.message_count > 20:
            print(f"Rate limit exceeded for {self.player_id}: {self.message_count} messages/second")
            return False
            
        return True

class ConnectionManager:
    def __init__(self):
        self.connections: Dict[str, ClientConnection] = {}
        self.game_state = GameState()
        
        # Start background tasks
        self.background_tasks = set()
        self.start_background_task(self.update_time_task())
        self.start_background_task(self.cleanup_inactive_players_task())
        self.start_background_task(self.keepalive_task())
        self.start_background_task(self.update_entities_task())
        self.start_background_task(self.broadcast_entities_task())
    
    def start_background_task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def connect(self, websocket: WebSocket) -> ClientConnection:
        """Handle a new client connection"""
        await websocket.accept()
        
        # Create new client connection
        client = ClientConnection(websocket, self)
        self.connections[client.player_id] = client
        print(f"New connection: {client.player_id}")
        
        # Authenticate client (get name)
        await client.authenticate()
        
        # Add player to game state
        self.game_state.add_player(client.player_id, client.name)
        
        # Get initial chunks for new player
        chunks = self.game_state.get_nearby_chunks(client.player_id)
        
        # Get nearby entities
        entities = self.game_state.get_nearby_entities(client.player_id)
        client.known_entities = set(entity['id'] for entity in entities)
        
        # Prepare player data for sending
        players_data = {}
        for pid, player_data in self.game_state.players.items():
            if pid != client.player_id:  # Don't include the new player
                players_data[pid] = {
                    'position': player_data['position'],
                    'rotation': player_data['rotation'],
                    'name': player_data['name']
                }
        
        # Send initial game state to new player
        await client.send_message({
            'type': 'init',
            'data': {
                'player_id': client.player_id,
                'chunks': chunks,
                'players': players_data,
                'entities': entities,
                'time_of_day': self.game_state.time_of_day
            }
        })
        
        # Notify other players about the new player
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
        """Handle client disconnection"""
        if client.player_id in self.connections:
            del self.connections[client.player_id]
            
        # Remove from game state
        self.game_state.remove_player(client.player_id)
        
        # Notify other clients
        await self.broadcast({
            'type': 'player_left',
            'data': {'player_id': client.player_id}
        })
        
        print(f"Disconnected: {client.player_id} ({client.name})")

    async def broadcast(self, message: Dict, exclude: Optional[str] = None):
        """Broadcast a message to all active clients except excluded one"""
        # Compress large messages once
        message_str = json.dumps(message)
        large_message = len(message_str) > 1024
        compressed = None
        
        if large_message:
            compressed = zlib.compress(message_str.encode(), 6)
        
        # Send to each client (make a copy to avoid issues with concurrent modifications)
        for player_id, client in list(self.connections.items()):
            if player_id != exclude and client.active:
                if large_message:
                    try:
                        await client.websocket.send_bytes(compressed)
                        self.game_state.messages_sent += 1
                    except Exception:
                        # Will be cleaned up by the client handler
                        client.active = False
                else:
                    try:
                        await client.websocket.send_text(message_str)
                        self.game_state.messages_sent += 1
                    except Exception:
                        # Will be cleaned up by the client handler
                        client.active = False

    async def update_player(self, client: ClientConnection, data: Dict):
        """Update player position and send nearby chunks"""
        position = data.get('position')
        rotation = data.get('rotation', 0)
        
        # Update game state
        self.game_state.update_player_position(client.player_id, position, rotation)
        
        # Get nearby chunks
        current_chunks = self.game_state.get_nearby_chunks(client.player_id)
        
        # Only send new chunks that weren't previously sent
        new_chunks = {}
        for key, chunk in current_chunks.items():
            if key not in client.sent_chunks:
                new_chunks[key] = chunk
        
        # Update sent chunks tracking
        client.sent_chunks.update(new_chunks.keys())
        
        # Get nearby entities
        entities = self.game_state.get_nearby_entities(client.player_id)
        entity_ids = set(entity['id'] for entity in entities)
        
        # Only send entities that are new or have been updated
        new_entities = [entity for entity in entities if entity['id'] not in client.known_entities]
        client.known_entities = entity_ids
        
        # Send chunks in small batches if there are many
        if new_chunks:
            chunk_keys = list(new_chunks.keys())
            for i in range(0, len(chunk_keys), 3):  # Send 3 chunks at a time
                batch = {k: new_chunks[k] for k in chunk_keys[i:i+3] if k in new_chunks}
                success = await client.send_message({
                    'type': 'chunks_update',
                    'data': {'chunks': batch}
                })
                
                if not success:
                    break  # Stop if send failed
                
                if i + 3 < len(chunk_keys):
                    await asyncio.sleep(0.05)  # Small delay between batches
        
        # Send new entities if any
        if new_entities:
            await client.send_message({
                'type': 'entities_update',
                'data': {'entities': new_entities}
            })
        
        # Broadcast position update to other players
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
        """Main handler for client connection"""
        client = await self.connect(websocket)
        
        try:
            while client.active:
                try:
                    # Wait for a message with a long timeout
                    # This allows us to catch disconnection on most platforms
                    data = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                    
                    # Update activity timestamp
                    client.last_activity = time.time()
                    self.game_state.messages_received += 1
                    
                    # Parse the message
                    message_data = None
                    try:
                        if "text" in data:
                            message_data = json.loads(data["text"])
                        elif "bytes" in data:
                            # Decompress binary messages
                            decompressed = zlib.decompress(data["bytes"])
                            message_data = json.loads(decompressed.decode())
                    except json.JSONDecodeError:
                        print(f"Error decoding message from {client.player_id}")
                        continue
                        
                    if not message_data:
                        continue
                    
                    # Check rate limiting
                    if not client.check_rate_limit():
                        await client.send_message({
                            'type': 'warning',
                            'data': {'message': 'Rate limit exceeded. Please slow down requests.'}
                        })
                        continue
                    
                    # Handle different message types
                    if message_data['type'] == 'position':
                        await self.update_player(client, message_data['data'])
                    elif message_data['type'] == 'ping':
                        await client.send_message({
                            'type': 'pong',
                            'data': {'server_time': time.time()}
                        })
                    elif message_data['type'] == 'interact_entity':
                        # Handle entity interaction (e.g., clicking or attacking)
                        entity_id = message_data['data'].get('entity_id')
                        action = message_data['data'].get('action')
                        if entity_id and action:
                            self.handle_entity_interaction(client, entity_id, action)
                
                except asyncio.TimeoutError:
                    # This is normal, just means no message received within timeout
                    continue
                    
                except WebSocketDisconnect:
                    # Client disconnected
                    print(f"WebSocket disconnected: {client.player_id}")
                    break
                    
                except Exception as e:
                    # Something went wrong
                    print(f"Error handling client {client.player_id}: {e}")
                    if "disconnect" in str(e).lower() or "closed" in str(e).lower():
                        break
        
        finally:
            # Ensure client is cleaned up
            await self.disconnect(client)
    
    def handle_entity_interaction(self, client, entity_id, action):
        """Handle player interaction with an entity"""
        # Find the entity
        entity = self.game_state.entity_manager.entities.get(entity_id)
        if not entity:
            return
            
        # Apply different actions
        if action == "click":
            # Make the entity respond in some way
            if isinstance(entity, RunningEntity):
                # Change direction dramatically
                angle = random.uniform(0, math.pi * 2)
                speed = math.sqrt(entity.velocity['x']**2 + entity.velocity['z']**2)
                entity.velocity['x'] = math.cos(angle) * speed * 1.5
                entity.velocity['z'] = math.sin(angle) * speed * 1.5
                entity.rotation = angle
            elif isinstance(entity, FlyingEntity):
                # Make it soar upward
                entity.velocity['y'] = random.uniform(5, 10)
            elif isinstance(entity, AbstractEntity):
                # Trigger teleportation
                entity.position['x'] += random.uniform(-10, 10)
                entity.position['y'] += random.uniform(5, 15)
                entity.position['z'] += random.uniform(-10, 10)

    async def update_time_task(self):
        """Background task to update time of day"""
        while True:
            try:
                # Update time of day
                self.game_state.time_of_day = (self.game_state.time_of_day + 1 / (self.game_state.DAY_NIGHT_CYCLE * 10)) % 1
                
                # Only broadcast time updates every second
                if int(self.game_state.time_of_day * self.game_state.DAY_NIGHT_CYCLE) % 10 == 0:
                    await self.broadcast({
                        'type': 'time_update',
                        'data': {'time_of_day': self.game_state.time_of_day}
                    })
                
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in update_time_task: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def cleanup_inactive_players_task(self):
        """Background task to remove inactive players"""
        while True:
            try:
                current_time = time.time()
                inactive_clients = []
                
                # Identify inactive clients
                for player_id, client in list(self.connections.items()):
                    if current_time - client.last_activity > 120:  # 2 minutes
                        inactive_clients.append(client)
                
                # Disconnect inactive clients
                for client in inactive_clients:
                    print(f"Removing inactive client: {client.player_id} ({client.name})")
                    await self.disconnect(client)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"Error in cleanup_inactive_players_task: {e}")
                await asyncio.sleep(30)

    async def keepalive_task(self):
        """Background task to send keepalive pings"""
        while True:
            try:
                current_time = time.time()
                
                # Send pings to clients that haven't sent messages recently
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
                
                await asyncio.sleep(15)  # Send keepalives every 15 seconds
            except Exception as e:
                print(f"Error in keepalive_task: {e}")
                await asyncio.sleep(15)
    
    async def update_entities_task(self):
        """Background task to update all entities"""
        while True:
            try:
                # Update entity positions and behaviors
                self.game_state.update_entities()
                await asyncio.sleep(0.05)  # 20 updates per second
            except Exception as e:
                print(f"Error in update_entities_task: {e}")
                await asyncio.sleep(1)
    
    async def broadcast_entities_task(self):
        """Background task to broadcast entity updates to clients"""
        while True:
            try:
                # Only broadcast every 100ms (10 times per second is enough for smooth visuals)
                await asyncio.sleep(0.1)
                
                # For each active client, send nearby entity updates
                for player_id, client in list(self.connections.items()):
                    if not client.active:
                        continue
                        
                    # Get current entities for this player
                    entities = self.game_state.get_nearby_entities(client.player_id)
                    
                    # Create update message with full entity data
                    if entities:
                        try:
                            await client.send_message({
                                'type': 'entities_update',
                                'data': {'entities': entities}
                            })
                        except Exception:
                            client.active = False
            except Exception as e:
                print(f"Error in broadcast_entities_task: {e}")
                await asyncio.sleep(1)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint - handles each client connection"""
    await manager.handle_client(websocket)

@app.get("/server-stats")
async def get_server_stats():
    """Endpoint to get server statistics"""
    return manager.game_state.get_stats()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
