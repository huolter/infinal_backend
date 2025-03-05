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
        self.position = position.copy()  # Use copy to avoid reference issues
        self.properties = properties or {}
        self.velocity = {'x': 0, 'y': 0, 'z': 0}
        self.rotation = 0
        self.created_at = time.time()
        self.last_update = self.created_at
        self.active = True

        # Set a long travel path
        self.path_length = random.uniform(500, 1000)  # How far the entity will travel
        self.distance_traveled = 0  # Track how far it has gone

        # Smoothing-related additions
        self.target_position = position.copy()  # Target position for smooth movement
        self.smoothing_factor = 0.1  # Controls the smoothness (smaller = smoother)

    def update(self, dt):
        """Update entity position based on velocity, with smoothing."""
        # Calculate target position
        self.target_position['x'] += self.velocity['x'] * dt
        self.target_position['y'] += self.velocity['y'] * dt
        self.target_position['z'] += self.velocity['z'] * dt

        # Calculate distance moved this frame (based on target position)
        distance_this_frame = math.sqrt(
            (self.velocity['x'] * dt) ** 2 +
            (self.velocity['z'] * dt) ** 2
        )
        self.distance_traveled += distance_this_frame

        # Smoothly interpolate towards the target position
        self.position['x'] += (self.target_position['x'] - self.position['x']) * self.smoothing_factor
        self.position['y'] += (self.target_position['y'] - self.position['y']) * self.smoothing_factor
        self.position['z'] += (self.target_position['z'] - self.position['z']) * self.smoothing_factor

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

    def is_expired(self):
        """Check if the entity has reached the end of its path"""
        return self.distance_traveled >= self.path_length

class HorizonTraveler(Entity):
    """Entity that travels from horizon to horizon"""
    def __init__(self, entity_id, position, world_center=None):
        entity_types = [
            'caravan', 'wanderer', 'traveler', 'nomad',
            'merchant', 'migrant', 'pilgrim', 'explorer'
        ]
        entity_type = random.choice(entity_types)

        if world_center is None:
            world_center = {'x': 0, 'z': 0}

        size = random.uniform(0.8, 2.5)
        color_hue = random.random()

        def hsl_to_rgb(h, s, l):
            if s == 0:
                r = g = b = l
            else:
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            return (int(r * 255), int(g * 255), int(b * 255))

        r, g, b = hsl_to_rgb(color_hue, 0.5, 0.6)
        color = f"#{r:02x}{g:02x}{b:02x}"

        properties = {
            'size': size,
            'color': color,
            'speed': random.uniform(0.2, 0.5),
            'isGiant': False,
            'companions': random.randint(0, 3),
            'startTime': time.time()
        }

        super().__init__(entity_id, entity_type, position, properties)

        travel_angle = random.uniform(0, math.pi * 2)
        horizon_distance = 100
        start_x = world_center['x'] + math.cos(travel_angle) * horizon_distance
        start_z = world_center['z'] + math.sin(travel_angle) * horizon_distance

        self.position = {'x': start_x, 'y': 0, 'z': start_z}

        end_x = world_center['x'] - math.cos(travel_angle) * horizon_distance
        end_z = world_center['z'] - math.sin(travel_angle) * horizon_distance

        dx = end_x - start_x
        dz = end_z - start_z
        distance = math.sqrt(dx*dx + dz*dz)

        speed = properties['speed']
        self.velocity = {
            'x': dx / distance * speed,
            'y': 0,
            'z': dz / distance * speed
        }

        self.rotation = math.atan2(dz, dx)
        self.path_length = distance
        self.walk_cycle = 0

    def update(self, dt):
        """Update horizon traveler with smooth, consistent movement"""
        self.walk_cycle += dt * self.properties['speed'] * 2
        self.position['y'] = max(0, math.sin(self.walk_cycle) * 0.05)
        super().update(dt)

class GiantTraveler(HorizonTraveler):
    """Giant entity that slowly walks from horizon to horizon"""
    def __init__(self, entity_id, position, world_center=None):
        super().__init__(entity_id, position, world_center)

        giant_types = ['giant', 'colossus', 'titan', 'behemoth', 'golem']
        self.type = random.choice(giant_types)

        giant_size = random.uniform(15.0, 30.0)
        color_hue = random.uniform(0.05, 0.15)

        def hsl_to_rgb(h, s, l):
            if s == 0:
                r = g = b = l
            else:
                def hue_to_rgb(p, q, t):
                    if t < 0: t += 1
                    if t > 1: t -= 1
                    if t < 1/6: return p + (q - p) * 6 * t
                    if t < 1/2: return q
                    if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                    return p
                q = l * (1 + s) if l < 0.5 else l + s - l * s
                p = 2 * l - q
                r = hue_to_rgb(p, q, h + 1/3)
                g = hue_to_rgb(p, q, h)
                b = hue_to_rgb(p, q, h - 1/3)
            return (int(r * 255), int(g * 255), int(b * 255))

        r, g, b = hsl_to_rgb(color_hue, 0.6, 0.4)
        giant_color = f"#{r:02x}{g:02x}{b:02x}"

        self.properties.update({
            'size': giant_size,
            'color': giant_color,
            'speed': random.uniform(0.05, 0.1),
            'isGiant': True,
            'companions': 0,
            'stepInterval': random.uniform(3.0, 5.0),
            'headSize': random.uniform(1.0, 1.5),
            'armLength': random.uniform(0.8, 1.2)
        })

        speed = self.properties['speed']
        direction = math.atan2(self.velocity['z'], self.velocity['x'])
        self.velocity = {
            'x': math.cos(direction) * speed,
            'y': 0,
            'z': math.sin(direction) * speed
        }

        self.position['y'] = giant_size * 0.6
        self.path_length *= 1.5
        self.last_footstep = time.time()
        self.footstep_interval = self.properties['stepInterval']

    def update(self, dt):
        """Update giant entity with slow, deliberate movements"""
        current_time = time.time()
        if current_time - self.last_footstep > self.footstep_interval:
            self.properties['footstep'] = True
            self.last_footstep = current_time
        else:
            self.properties['footstep'] = False

        sway = math.sin(current_time * 0.2) * 0.02
        self.position['y'] = self.properties['size'] * 0.6 + sway
        Entity.update(self, dt)

class SkyTraveler(Entity):
    """Entity that glides slowly across the sky"""
    def __init__(self, entity_id, position, world_center=None):
        sky_types = ['balloon', 'cloud', 'airship', 'floater',
                     'glider', 'zeppelin', 'blimp', 'kite']
        entity_type = random.choice(sky_types)

        if world_center is None:
            world_center = {'x': 0, 'z': 0}

        size = random.uniform(4.0, 10.0)

        def generate_pastel():
            r = random.randint(180, 255)
            g = random.randint(180, 255)
            b = random.randint(180, 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        color = generate_pastel()

        properties = {
            'size': size,
            'color': color,
            'speed': random.uniform(0.1, 0.3),
            'altitude': random.uniform(20, 50),
            'startTime': time.time()
        }

        super().__init__(entity_id, entity_type, position, properties)

        travel_angle = random.uniform(0, math.pi * 2)
        horizon_distance = 100
        start_x = world_center['x'] + math.cos(travel_angle) * horizon_distance
        start_z = world_center['z'] + math.sin(travel_angle) * horizon_distance

        self.position = {'x': start_x, 'y': properties['altitude'], 'z': start_z}

        end_x = world_center['x'] - math.cos(travel_angle) * horizon_distance
        end_z = world_center['z'] - math.sin(travel_angle) * horizon_distance

        dx = end_x - start_x
        dz = end_z - start_z
        distance = math.sqrt(dx*dx + dz*dz)

        speed = properties['speed']
        self.velocity = {
            'x': dx / distance * speed,
            'y': 0,
            'z': dz / distance * speed
        }

        self.rotation = math.atan2(dz, dx)
        self.path_length = distance

    def update(self, dt):
        """Update sky traveler with smooth drifting motion"""
        current_time = time.time()
        time_factor = current_time - self.properties['startTime']
        vertical_offset = math.sin(time_factor * 0.1) * 0.3
        self.position['y'] = self.properties['altitude'] + vertical_offset
        super().update(dt)

class ChunkGenerator:
    def __init__(self, entity_manager):
        self.chunks = LRUCache(maxsize=2000)
        self.DENSITY = 0.3
        self.seed = random.randint(0, 1000000)
        self.biome_scale = 100.0
        self.terrain_scale = 50.0
        self.entity_manager = entity_manager

        self.generated_chunks = 0
        self.cache_hits = 0

        self.last_horizon_spawn = time.time() - 999
        self.last_sky_spawn = time.time() - 999
        self.last_giant_spawn = time.time() - 999

        self.horizon_travelers_active = 0
        self.sky_travelers_active = 0
        self.giants_active = 0

        self.MAX_HORIZON_TRAVELERS = 20
        self.MAX_SKY_TRAVELERS = 15
        self.MAX_GIANTS = 5

        self.next_horizon_spawn = 0
        self.next_sky_spawn = 5
        self.next_giant_spawn = 10

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

        chunk['static_entities'] = self.generate_static_entities(chunk_x, chunk_z, biome)

        if biome == 'plains' and random.random() < 0.1:
            water_size = random.randint(3, 6)
            water_x = random.randint(0, 15 - water_size)
            water_z = random.randint(0, 15 - water_size)
            chunk['water'] = {'size': water_size, 'position': {'x': water_x, 'z': water_z}}

        current_time = time.time()
        world_center = {'x': 0, 'z': 0}

        chunk_center_x = chunk_x * 16 + 8
        chunk_center_z = chunk_z * 16 + 8
        distance_from_center = math.sqrt(chunk_center_x**2 + chunk_center_z**2)

        if distance_from_center < 800:
            if (current_time > self.next_horizon_spawn and
                self.horizon_travelers_active < self.MAX_HORIZON_TRAVELERS):
                for _ in range(random.randint(1, 3)):
                    self.entity_manager.create_horizon_traveler(world_center)
                    self.horizon_travelers_active += 1
                self.next_horizon_spawn = current_time + random.uniform(15, 30)

            if (current_time > self.next_sky_spawn and
                self.sky_travelers_active < self.MAX_SKY_TRAVELERS):
                for _ in range(random.randint(1, 2)):
                    self.entity_manager.create_sky_traveler(world_center)
                    self.sky_travelers_active += 1
                self.next_sky_spawn = current_time + random.uniform(20, 40)

            if (current_time > self.next_giant_spawn and
                self.giants_active < self.MAX_GIANTS):
                self.entity_manager.create_giant_traveler(world_center)
                self.giants_active += 1
                self.next_giant_spawn = current_time + random.uniform(40, 80)

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

    def generate_static_entities(self, chunk_x: int, chunk_z: int, biome: str) -> List[Dict]:
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
            "cache_size": len(self.chunks),
            "horizon_travelers": self.horizon_travelers_active,
            "sky_travelers": self.sky_travelers_active,
            "giants": self.giants_active
        }

    def update_entity_counts(self, entity_counts):
        """Update the counts of active special entities"""
        self.horizon_travelers_active = entity_counts.get('horizon', 0)
        self.sky_travelers_active = entity_counts.get('sky', 0)
        self.giants_active = entity_counts.get('giant', 0)

        current_time = time.time()
        if self.horizon_travelers_active == 0 and current_time > self.next_horizon_spawn:
            world_center = {'x': 0, 'z': 0}
            for _ in range(random.randint(2, 5)):
                self.entity_manager.create_horizon_traveler(world_center)
                self.horizon_travelers_active += 1
            self.next_horizon_spawn = current_time + random.uniform(10, 20)

        if self.sky_travelers_active == 0 and current_time > self.next_sky_spawn:
            world_center = {'x': 0, 'z': 0}
            for _ in range(random.randint(2, 3)):
                self.entity_manager.create_sky_traveler(world_center)
                self.sky_travelers_active += 1
            self.next_sky_spawn = current_time + random.uniform(15, 30)

        if self.giants_active == 0 and current_time > self.next_giant_spawn:
            world_center = {'x': 0, 'z': 0}
            self.entity_manager.create_giant_traveler(world_center)
            self.giants_active += 1
            self.next_giant_spawn = current_time + random.uniform(30, 60)

class EntityManager:
    """Manages all dynamic entities in the game"""
    def __init__(self):
        self.entities = {}
        self.entity_counter = 0
        self.last_update = time.time()

    def create_horizon_traveler(self, world_center):
        """Create a new horizon traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        entity = HorizonTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity

        num_companions = entity.properties.get('companions', 0)
        for i in range(num_companions):
            self.entity_counter += 1
            companion_id = f"entity_{self.entity_counter}"
            offset_x = random.uniform(-5, 5)
            offset_z = random.uniform(-5, 5)
            companion_pos = {
                'x': entity.position['x'] + offset_x,
                'y': 0,
                'z': entity.position['z'] + offset_z
            }
            companion = HorizonTraveler(companion_id, companion_pos, world_center)
            companion.velocity = entity.velocity.copy()
            companion.rotation = entity.rotation
            companion.properties['size'] *= 0.8
            companion.properties['companions'] = 0
            self.entities[companion_id] = companion

        print(f"Created horizon traveler {entity.type} with {num_companions} companions")
        return entity

    def create_sky_traveler(self, world_center):
        """Create a new sky traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        entity = SkyTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity
        print(f"Created sky traveler {entity.type} at altitude {entity.properties['altitude']:.1f}")
        return entity

    def create_giant_traveler(self, world_center):
        """Create a new giant traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        entity = GiantTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity
        print(f"Created giant {entity.type} of size {entity.properties['size']:.1f}")
        return entity

    def update(self, dt, chunk_generator=None):
        """Update all entities"""
        entity_counts = {'horizon': 0, 'sky': 0, 'giant': 0}
        entities_to_remove = []

        for entity_id, entity in self.entities.items():
            try:
                entity.update(dt)
                if isinstance(entity, GiantTraveler):
                    entity_counts['giant'] += 1
                elif isinstance(entity, SkyTraveler):
                    entity_counts['sky'] += 1
                elif isinstance(entity, HorizonTraveler) and not isinstance(entity, GiantTraveler):
                    entity_counts['horizon'] += 1
                if entity.is_expired():
                    entities_to_remove.append(entity_id)
            except Exception as e:
                print(f"Error updating entity {entity_id}: {e}")
                entities_to_remove.append(entity_id)

        for entity_id in entities_to_remove:
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entity_type = "unknown"
                if isinstance(entity, GiantTraveler):
                    entity_type = "giant"
                elif isinstance(entity, SkyTraveler):
                    entity_type = "sky"
                elif isinstance(entity, HorizonTraveler):
                    entity_type = "horizon"
                print(f"Removing {entity_type} entity {entity_id} ({entity.type}) - path completed")
                del self.entities[entity_id]
                if entity_type in entity_counts:
                    entity_counts[entity_type] -= 1

        if chunk_generator:
            chunk_generator.update_entity_counts(entity_counts)

    def get_entities_in_range(self, position, radius):
        """Get all entities within a certain radius of a position"""
        entities_in_range = []
        for entity in self.entities.values():
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
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        return {
            "total_entities": len(self.entities),
            "entity_types": entity_types
        }

class GameState:
    def __init__(self):
        self.entity_manager = EntityManager()
        self.players: Dict[str, Dict] = {}
        self.chunk_generator = ChunkGenerator(self.entity_manager)
        self.VIEW_DISTANCE = 3
        self.time_of_day = 0
        self.DAY_NIGHT_CYCLE = 600
        self.last_activity = {}
        self.connections_total = 0
        self.connections_active = 0
        self.messages_received = 0
        self.messages_sent = 0
        self.last_entity_update = time.time()
        self.initial_entities_spawned = False

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

        if not self.initial_entities_spawned:
            self.initial_entities_spawned = True
            world_center = {'x': 0, 'z': 0}
            for _ in range(10):
                self.entity_manager.create_horizon_traveler(world_center)
            for _ in range(5):
                self.entity_manager.create_sky_traveler(world_center)
            for _ in range(2):
                self.entity_manager.create_giant_traveler(world_center)
            print("Initial entities spawned")

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

    def get_nearby_entities(self, player_id: str) -> List[Dict]:
        """Get all dynamic entities near a player"""
        if player_id not in self.players:
            return []
        pos = self.players[player_id]['position']
        radius = self.VIEW_DISTANCE * 32
        return self.entity_manager.get_entities_in_range(pos, radius)

    def update_entities(self):
        """Update all dynamic entities"""
        current_time = time.time()
        if current_time - self.last_entity_update >= 0.1:
            dt = current_time - self.last_entity_update
            self.entity_manager.update(dt, self.chunk_generator)
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
        self.known_entities = set()
        self.message_count = 0
        self.message_time = time.time()

    async def authenticate(self, timeout=5.0):
        """Try to get player name from authentication message"""
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
        """Send a message to this client with error handling"""
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
        """Check if client is sending too many messages"""
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
    def __init__(self):
        self.connections: Dict[str, ClientConnection] = {}
        self.game_state = GameState()
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
        client = ClientConnection(websocket, self)
        self.connections[client.player_id] = client
        print(f"New connection: {client.player_id}")

        await client.authenticate()
        self.game_state.add_player(client.player_id, client.name)

        chunks = self.game_state.get_nearby_chunks(client.player_id)
        entities = self.game_state.get_nearby_entities(client.player_id)
        client.known_entities = set(entity['id'] for entity in entities)

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
                'entities': entities,
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
        """Handle client disconnection"""
        if client.player_id in self.connections:
            del self.connections[client.player_id]
        self.game_state.remove_player(client.player_id)
        await self.broadcast({
            'type': 'player_left',
            'data': {'player_id': client.player_id}
        })
        print(f"Disconnected: {client.player_id} ({client.name})")

    async def broadcast(self, message: Dict, exclude: Optional[str] = None):
        """Broadcast a message to all active clients except excluded one"""
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
        """Update player position and send nearby chunks"""
        position = data.get('position')
        rotation = data.get('rotation', 0)
        self.game_state.update_player_position(client.player_id, position, rotation)

        current_chunks = self.game_state.get_nearby_chunks(client.player_id)
        new_chunks = {k: v for k, v in current_chunks.items() if k not in client.sent_chunks}
        client.sent_chunks.update(new_chunks.keys())

        entities = self.game_state.get_nearby_entities(client.player_id)
        entity_ids = set(entity['id'] for entity in entities)
        new_entities = [entity for entity in entities if entity['id'] not in client.known_entities]
        client.known_entities = entity_ids

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

        if new_entities:
            await client.send_message({
                'type': 'entities_update',
                'data': {'entities': new_entities}
            })

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
                    elif message_data['type'] == 'interact_entity':
                        entity_id = message_data['data'].get('entity_id')
                        action = message_data['data'].get('action')
                        if entity_id and action:
                            self.handle_entity_interaction(client, entity_id, action)

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

    def handle_entity_interaction(self, client, entity_id, action):
        """Handle player interaction with an entity"""
        entity = self.game_state.entity_manager.entities.get(entity_id)
        if not entity:
            return
        if action == "click" and isinstance(entity, GiantTraveler):
            player_pos = self.game_state.players[client.player_id]['position']
            dx = player_pos['x'] - entity.position['x']
            dz = player_pos['z'] - entity.position['z']
            entity.rotation = math.atan2(dz, dx)
            entity.properties['waving'] = True
            asyncio.create_task(self.reset_giant_wave(entity_id))

    async def reset_giant_wave(self, entity_id):
        """Reset the giant's waving flag after a delay"""
        await asyncio.sleep(3)
        entity = self.game_state.entity_manager.entities.get(entity_id)
        if entity and isinstance(entity, GiantTraveler):
            entity.properties['waving'] = False

    async def update_time_task(self):
        """Background task to update time of day"""
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

    async def cleanup_inactive_players_task(self):
        """Background task to remove inactive players"""
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
        """Background task to send keepalive pings"""
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

    async def update_entities_task(self):
        """Background task to update all entities"""
        while True:
            try:
                self.game_state.update_entities()
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in update_entities_task: {e}")
                await asyncio.sleep(1)

    async def broadcast_entities_task(self):
        """Background task to broadcast entity updates to clients"""
        while True:
            try:
                await asyncio.sleep(0.1)  # Changed from 0.2 to 0.1 for 10 updates per second
                for player_id, client in list(self.connections.items()):
                    if not client.active:
                        continue
                    entities = self.game_state.get_nearby_entities(client.player_id)
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