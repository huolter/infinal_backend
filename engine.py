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
        
    def update(self, dt):
        """Update entity position based on velocity"""
        # Calculate distance moved this frame
        distance_this_frame = math.sqrt(
            (self.velocity['x'] * dt) ** 2 +
            (self.velocity['z'] * dt) ** 2
        )
        
        # Update position
        self.position['x'] += self.velocity['x'] * dt
        self.position['y'] += self.velocity['y'] * dt
        self.position['z'] += self.velocity['z'] * dt
        
        # Track total distance traveled
        self.distance_traveled += distance_this_frame
        
        # Update last update time
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
        # The types of entity that can appear on the horizon
        entity_types = [
            'caravan', 'wanderer', 'traveler', 'nomad', 
            'merchant', 'migrant', 'pilgrim', 'explorer'
        ]
        
        # Randomly select a type
        entity_type = random.choice(entity_types)
        
        # Base position - will be overridden by horizon placement
        if world_center is None:
            world_center = {'x': 0, 'z': 0}
            
        # Generate random properties
        size = random.uniform(0.8, 2.5)
        color_hue = random.random()  # Random hue (0-1)
        
        # Convert HSL to RGB for a nice consistent color
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
        
        # Generate a pleasing muted color
        r, g, b = hsl_to_rgb(color_hue, 0.5, 0.6)
        color = f"#{r:02x}{g:02x}{b:02x}"
        
        properties = {
            'size': size,
            'color': color,
            'speed': random.uniform(0.2, 0.5),  # Slow, consistent speed
            'isGiant': False,
            'companions': random.randint(0, 3),  # How many additional entities travel with this one
            'startTime': time.time()
        }
        
        super().__init__(entity_id, entity_type, position, properties)
        
        # Set up path that crosses the world
        # First, calculate a random angle through the world center
        travel_angle = random.uniform(0, math.pi * 2)
        
        # Calculate a starting point far on the horizon (beyond view distance)
        horizon_distance = 300  # Well beyond view distance
        start_x = world_center['x'] + math.cos(travel_angle) * horizon_distance
        start_z = world_center['z'] + math.sin(travel_angle) * horizon_distance
        
        # Set starting position at the horizon
        self.position = {
            'x': start_x,
            'y': 0,  # Will be set to ground level
            'z': start_z
        }
        
        # Calculate end point on the opposite horizon
        end_x = world_center['x'] - math.cos(travel_angle) * horizon_distance
        end_z = world_center['z'] - math.sin(travel_angle) * horizon_distance
        
        # Calculate direction vector
        dx = end_x - start_x
        dz = end_z - start_z
        distance = math.sqrt(dx*dx + dz*dz)
        
        # Normalize direction vector and scale by speed
        speed = properties['speed']
        self.velocity = {
            'x': dx / distance * speed,
            'y': 0,
            'z': dz / distance * speed
        }
        
        # Set rotation to face the direction of travel
        self.rotation = math.atan2(dz, dx)
        
        # Set path length to the total horizon-to-horizon distance
        self.path_length = distance
        
        # For smooth walking animation (very subtle)
        self.walk_cycle = 0
        
    def update(self, dt):
        """Update horizon traveler with smooth, consistent movement"""
        # Increment walk cycle
        self.walk_cycle += dt * self.properties['speed'] * 2
        
        # Very subtle height variation for walking
        self.position['y'] = max(0, math.sin(self.walk_cycle) * 0.05)
        
        # Apply base update to move along the path
        super().update(dt)

class GiantTraveler(HorizonTraveler):
    """Giant entity that slowly walks from horizon to horizon"""
    def __init__(self, entity_id, position, world_center=None):
        # First initialize as a normal traveler
        super().__init__(entity_id, position, world_center)
        
        # Override with giant-specific properties
        giant_types = [
            'giant', 'colossus', 'titan', 'behemoth', 'golem'
        ]
        
        self.type = random.choice(giant_types)
        
        # Giant-specific properties
        giant_size = random.uniform(10.0, 20.0)  # 10-20x normal size
        
        # Generate a more muted, earthy color for giants
        color_hue = random.uniform(0.05, 0.15)  # Brown/amber hues
        
        # Convert HSL to RGB
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
        
        # Override properties
        self.properties.update({
            'size': giant_size,
            'color': giant_color,
            'speed': random.uniform(0.05, 0.1),  # Much slower than normal travelers
            'isGiant': True,
            'companions': 0,  # Giants travel alone
            'stepInterval': random.uniform(3.0, 5.0),  # Time between footsteps
            'headSize': random.uniform(1.0, 1.5),
            'armLength': random.uniform(0.8, 1.2)
        })
        
        # Adjust velocity for slower speed
        speed = self.properties['speed']
        direction = math.atan2(self.velocity['z'], self.velocity['x'])
        
        self.velocity = {
            'x': math.cos(direction) * speed,
            'y': 0,
            'z': math.sin(direction) * speed
        }
        
        # Add head position offset for height
        self.position['y'] = giant_size * 0.6  # Base height is roughly 60% of size
        
        # Set longer path to stay visible longer
        self.path_length *= 1.5
        
        # Footstep tracking
        self.last_footstep = time.time()
        self.footstep_interval = self.properties['stepInterval']
        
    def update(self, dt):
        """Update giant entity with slow, deliberate movements"""
        current_time = time.time()
        
        # Handle footsteps
        if current_time - self.last_footstep > self.footstep_interval:
            # Add very subtle footstep effect
            self.properties['footstep'] = True  # Signal to client for footstep effect
            self.last_footstep = current_time
        else:
            self.properties['footstep'] = False
        
        # Almost imperceptible gentle swaying for realism
        sway = math.sin(current_time * 0.2) * 0.02
        self.position['y'] = self.properties['size'] * 0.6 + sway
            
        # Call parent update to apply velocity (skip HorizonTraveler update)
        Entity.update(self, dt)

class SkyTraveler(Entity):
    """Entity that glides slowly across the sky"""
    def __init__(self, entity_id, position, world_center=None):
        # Types of sky travelers
        sky_types = [
            'balloon', 'cloud', 'airship', 'floater', 
            'glider', 'zeppelin', 'blimp', 'kite'
        ]
        
        # Randomly select a type
        entity_type = random.choice(sky_types)
        
        # Base position - will be overridden by horizon placement
        if world_center is None:
            world_center = {'x': 0, 'z': 0}
            
        # Generate random properties
        size = random.uniform(2.0, 6.0)
        
        # Generate a soft, pastel color
        def generate_pastel():
            r = random.randint(180, 255)
            g = random.randint(180, 255)
            b = random.randint(180, 255)
            return f"#{r:02x}{g:02x}{b:02x}"
            
        color = generate_pastel()
        
        properties = {
            'size': size,
            'color': color,
            'speed': random.uniform(0.1, 0.3),  # Very slow, drifting speed
            'altitude': random.uniform(30, 80),  # Height above ground
            'startTime': time.time()
        }
        
        super().__init__(entity_id, entity_type, position, properties)
        
        # Set up path that crosses the sky
        # Calculate a random angle through the world center
        travel_angle = random.uniform(0, math.pi * 2)
        
        # Calculate a starting point far on the horizon (beyond view distance)
        horizon_distance = 250  # Well beyond view distance
        start_x = world_center['x'] + math.cos(travel_angle) * horizon_distance
        start_z = world_center['z'] + math.sin(travel_angle) * horizon_distance
        
        # Set starting position at the horizon in the sky
        self.position = {
            'x': start_x,
            'y': properties['altitude'],
            'z': start_z
        }
        
        # Calculate end point on the opposite horizon
        end_x = world_center['x'] - math.cos(travel_angle) * horizon_distance
        end_z = world_center['z'] - math.sin(travel_angle) * horizon_distance
        
        # Calculate direction vector
        dx = end_x - start_x
        dz = end_z - start_z
        distance = math.sqrt(dx*dx + dz*dz)
        
        # Normalize direction vector and scale by speed
        speed = properties['speed']
        self.velocity = {
            'x': dx / distance * speed,
            'y': 0,  # Level flight
            'z': dz / distance * speed
        }
        
        # Set rotation to face the direction of travel
        self.rotation = math.atan2(dz, dx)
        
        # Set path length to the total horizon-to-horizon distance
        self.path_length = distance
        
    def update(self, dt):
        """Update sky traveler with smooth drifting motion"""
        current_time = time.time()
        
        # Add very gentle vertical oscillation (barely noticeable bobbing)
        time_factor = current_time - self.properties['startTime']
        vertical_offset = math.sin(time_factor * 0.1) * 0.3
        
        # Adjust height with the subtle oscillation
        self.position['y'] = self.properties['altitude'] + vertical_offset
        
        # Apply base update to continue along the path
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
        
        # Track when we last spawned special entities
        self.last_horizon_spawn = time.time() - 60  # Start with a delay
        self.last_sky_spawn = time.time() - 30  # Stagger the spawns
        self.last_giant_spawn = time.time() - 90  # Even longer initial delay for giants
        
        # Count of active special entities
        self.horizon_travelers_active = 0
        self.sky_travelers_active = 0
        self.giants_active = 0
        
        # Maximum number of each type to have active at once
        self.MAX_HORIZON_TRAVELERS = 3
        self.MAX_SKY_TRAVELERS = 2
        self.MAX_GIANTS = 1

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
            
        # Consider spawning special horizon/sky travelers when generating chunks
        # We only want to do this occasionally to avoid too many entities
        current_time = time.time()
        world_center = {'x': 0, 'z': 0}  # Assume world center is at origin
        
        # Only check for spawning if this chunk is within a reasonable distance of the center
        # This avoids spawning entities too far away when players explore distant areas
        chunk_center_x = chunk_x * 16 + 8
        chunk_center_z = chunk_z * 16 + 8
        distance_from_center = math.sqrt(chunk_center_x**2 + chunk_center_z**2)
        
        if distance_from_center < 500:  # Within reasonable distance of center
            # Only spawn travelers occasionally based on timers
            if (current_time - self.last_horizon_spawn > 120 and  # 2 minutes between spawns
                self.horizon_travelers_active < self.MAX_HORIZON_TRAVELERS):
                
                # Spawn a new horizon traveler
                self.entity_manager.create_horizon_traveler(world_center)
                self.last_horizon_spawn = current_time
                self.horizon_travelers_active += 1
                
            if (current_time - self.last_sky_spawn > 180 and  # 3 minutes between spawns
                self.sky_travelers_active < self.MAX_SKY_TRAVELERS):
                
                # Spawn a new sky traveler
                self.entity_manager.create_sky_traveler(world_center)
                self.last_sky_spawn = current_time
                self.sky_travelers_active += 1
                
            if (current_time - self.last_giant_spawn > 300 and  # 5 minutes between giants
                self.giants_active < self.MAX_GIANTS):
                
                # Spawn a new giant
                self.entity_manager.create_giant_traveler(world_center)
                self.last_giant_spawn = current_time
                self.giants_active += 1

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
            "horizon_travelers": self.horizon_travelers_active,
            "sky_travelers": self.sky_travelers_active,
            "giants": self.giants_active
        }
        
    def update_entity_counts(self, entity_counts):
        """Update the counts of active special entities"""
        self.horizon_travelers_active = entity_counts.get('horizon', 0)
        self.sky_travelers_active = entity_counts.get('sky', 0)
        self.giants_active = entity_counts.get('giant', 0)

class EntityManager:
    """Manages all dynamic entities in the game"""
    def __init__(self):
        self.entities = {}  # id -> entity
        self.entity_counter = 0
        self.last_update = time.time()
        
    def create_horizon_traveler(self, world_center):
        """Create a new horizon traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        
        # Create the traveler with the current world center
        entity = HorizonTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity
        
        # If the traveler has companions, create them too
        num_companions = entity.properties.get('companions', 0)
        for i in range(num_companions):
            self.entity_counter += 1
            companion_id = f"entity_{self.entity_counter}"
            
            # Create companion with slight offset
            offset_x = random.uniform(-5, 5)
            offset_z = random.uniform(-5, 5)
            companion_pos = {
                'x': entity.position['x'] + offset_x,
                'y': 0,
                'z': entity.position['z'] + offset_z
            }
            
            companion = HorizonTraveler(companion_id, companion_pos, world_center)
            
            # Make sure companions have the same speed and direction
            companion.velocity = entity.velocity.copy()
            companion.rotation = entity.rotation
            
            # Adjust companion properties
            companion.properties['size'] *= 0.8  # Slightly smaller
            companion.properties['companions'] = 0  # No further companions
            
            self.entities[companion_id] = companion
        
        print(f"Created horizon traveler {entity.type} with {num_companions} companions")
        return entity
        
    def create_sky_traveler(self, world_center):
        """Create a new sky traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        
        # Create the traveler with the current world center
        entity = SkyTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity
        
        print(f"Created sky traveler {entity.type} at altitude {entity.properties['altitude']:.1f}")
        return entity
        
    def create_giant_traveler(self, world_center):
        """Create a new giant traveler entity"""
        self.entity_counter += 1
        entity_id = f"entity_{self.entity_counter}"
        
        # Create the giant with the current world center
        entity = GiantTraveler(entity_id, {'x': 0, 'y': 0, 'z': 0}, world_center)
        self.entities[entity_id] = entity
        
        print(f"Created giant {entity.type} of size {entity.properties['size']:.1f}")
        return entity
        
    def update(self, dt, chunk_generator=None):
        """Update all entities"""
        # Track counts of special entity types
        entity_counts = {
            'horizon': 0,
            'sky': 0,
            'giant': 0
        }
        
        # List of entities to remove
        entities_to_remove = []
        
        # Update each entity
        for entity_id, entity in self.entities.items():
            try:
                # Apply movement update
                entity.update(dt)
                
                # Count special entity types
                if isinstance(entity, GiantTraveler):
                    entity_counts['giant'] += 1
                elif isinstance(entity, SkyTraveler):
                    entity_counts['sky'] += 1
                elif isinstance(entity, HorizonTraveler):
                    entity_counts['horizon'] += 1
                
                # Check if entity has reached the end of its path (expired)
                if entity.is_expired():
                    entities_to_remove.append(entity_id)
                    
            except Exception as e:
                print(f"Error updating entity {entity_id}: {e}")
                entities_to_remove.append(entity_id)
                
        # Remove entities that have reached the end of their path
        for entity_id in entities_to_remove:
            if entity_id in self.entities:
                # Get the entity type before removing
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
                
                # Adjust count since we just removed one
                if entity_type in entity_counts:
                    entity_counts[entity_type] -= 1
                    
        # Update chunk generator's counts if provided
        if chunk_generator:
            chunk_generator.update_entity_counts(entity_counts)
            
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
        
        # Use a much larger radius for entities to ensure smooth transitions
        radius = self.VIEW_DISTANCE * 32  # Double the normal view radius
        
        # Get entities within range
        return self.entity_manager.get_entities_in_range(pos, radius)
    
    def update_entities(self):
        """Update all dynamic entities"""
        current_time = time.time()
        if current_time - self.last_entity_update >= 0.1:  # 10 updates per second
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
            # Wave at the player if it's a giant
            if isinstance(entity, GiantTraveler):
                # Make the giant turn toward the player and wave
                player_pos = self.game_state.players[client.player_id]['position']
                dx = player_pos['x'] - entity.position['x']
                dz = player_pos['z'] - entity.position['z']
                entity.rotation = math.atan2(dz, dx)
                
                # Set a 'waving' flag for the client to animate
                entity.properties['waving'] = True
                
                # Reset waving after 3 seconds
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
                await asyncio.sleep(0.1)  # 10 updates per second
            except Exception as e:
                print(f"Error in update_entities_task: {e}")
                await asyncio.sleep(1)
    
    async def broadcast_entities_task(self):
        """Background task to broadcast entity updates to clients"""
        while True:
            try:
                # Broadcast entity updates 5 times per second
                await asyncio.sleep(0.2)
                
                # For each active client, send nearby entity updates
                for player_id, client in list(self.connections.items()):
                    if not client.active:
                        continue
                        
                    # Get current entities for this player
                    entities = self.game_state.get_nearby_entities(client.player_id)
                    
                    # Only send updates if there are entities to update
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
