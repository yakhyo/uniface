#!/usr/bin/env python3
"""
Persistent Face ID Tracking with Webcam

This script assigns a unique, persistent ID to each person detected by the webcam.
If a person leaves and returns, they will be recognized and assigned the same ID.

Supports multiple storage backends:
  - memory: In-memory only (no persistence)
  - file: NumPy .npz file (default)
  - redis: Redis database
  - mongodb: MongoDB database

Usage: 
    python run_face_tracking.py
    python run_face_tracking.py --storage file --save-db faces.npz
    python run_face_tracking.py --storage redis --redis-url redis://localhost:6379
    python run_face_tracking.py --storage mongodb --mongo-url mongodb://localhost:27017
"""

import argparse
import base64
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from uniface import RetinaFace, SCRFD
from uniface.recognition import ArcFace, MobileFace, SphereFace
from uniface.face_utils import compute_similarity


@dataclass
class TrackedFace:
    """Represents a tracked face with its embedding and metadata."""
    face_id: int
    embedding: np.ndarray
    last_seen: float = field(default_factory=time.time)
    seen_count: int = 1
    name: Optional[str] = None  # Optional name label


# =============================================================================
# Storage Backends
# =============================================================================

class StorageBackend(ABC):
    """Abstract base class for face database storage backends."""
    
    @abstractmethod
    def load_all(self) -> Dict[int, TrackedFace]:
        """Load all faces from storage."""
        pass
    
    @abstractmethod
    def save_face(self, face: TrackedFace) -> None:
        """Save or update a single face."""
        pass
    
    @abstractmethod
    def delete_face(self, face_id: int) -> None:
        """Delete a face by ID."""
        pass
    
    @abstractmethod
    def get_next_id(self) -> int:
        """Get the next available face ID."""
        pass
    
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all faces from storage."""
        pass
    
    def close(self) -> None:
        """Close any connections (optional)."""
        pass


class MemoryStorage(StorageBackend):
    """In-memory storage (no persistence)."""
    
    def __init__(self):
        self.faces: Dict[int, TrackedFace] = {}
        self._next_id = 1
    
    def load_all(self) -> Dict[int, TrackedFace]:
        return self.faces.copy()
    
    def save_face(self, face: TrackedFace) -> None:
        self.faces[face.face_id] = face
        if face.face_id >= self._next_id:
            self._next_id = face.face_id + 1
    
    def delete_face(self, face_id: int) -> None:
        self.faces.pop(face_id, None)
    
    def get_next_id(self) -> int:
        next_id = self._next_id
        self._next_id += 1
        return next_id
    
    def clear_all(self) -> None:
        self.faces.clear()
        self._next_id = 1


class FileStorage(StorageBackend):
    """NumPy .npz file storage."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._next_id = 1
    
    def load_all(self) -> Dict[int, TrackedFace]:
        faces = {}
        if self.db_path.exists():
            try:
                data = np.load(self.db_path, allow_pickle=True)
                face_ids = data['face_ids']
                embeddings = data['embeddings']
                names = data.get('names', [None] * len(face_ids))
                
                for face_id, embedding, name in zip(face_ids, embeddings, names):
                    faces[int(face_id)] = TrackedFace(
                        face_id=int(face_id),
                        embedding=embedding,
                        name=name if name else None,
                    )
                
                if faces:
                    self._next_id = max(faces.keys()) + 1
                
                print(f"[FileStorage] Loaded {len(faces)} faces from {self.db_path}")
            except Exception as e:
                print(f"[FileStorage] Warning: Could not load database: {e}")
        return faces
    
    def _save_all(self, faces: Dict[int, TrackedFace]) -> None:
        """Save all faces to file."""
        if not faces:
            return
        
        face_ids = np.array(list(faces.keys()))
        embeddings = np.array([f.embedding for f in faces.values()])
        names = np.array([f.name or "" for f in faces.values()])
        
        np.savez(self.db_path, face_ids=face_ids, embeddings=embeddings, names=names)
    
    def save_face(self, face: TrackedFace) -> None:
        # Load existing, update, and save all
        faces = self.load_all()
        faces[face.face_id] = face
        self._save_all(faces)
        if face.face_id >= self._next_id:
            self._next_id = face.face_id + 1
    
    def delete_face(self, face_id: int) -> None:
        faces = self.load_all()
        faces.pop(face_id, None)
        self._save_all(faces)
    
    def get_next_id(self) -> int:
        next_id = self._next_id
        self._next_id += 1
        return next_id
    
    def clear_all(self) -> None:
        if self.db_path.exists():
            self.db_path.unlink()
        self._next_id = 1


class RedisStorage(StorageBackend):
    """Redis storage backend for real-time persistence."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "uniface"):
        try:
            import redis
        except ImportError:
            raise ImportError("Redis support requires: pip install redis")
        
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.prefix = prefix
        self._faces_key = f"{prefix}:faces"
        self._counter_key = f"{prefix}:next_id"
        
        # Test connection
        try:
            self.redis.ping()
            print(f"[RedisStorage] Connected to {redis_url}")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to Redis at {redis_url}: {e}")
    
    def _serialize_face(self, face: TrackedFace) -> bytes:
        """Serialize a TrackedFace to bytes."""
        data = {
            'face_id': face.face_id,
            'embedding': base64.b64encode(face.embedding.tobytes()).decode('ascii'),
            'embedding_shape': face.embedding.shape,
            'last_seen': face.last_seen,
            'seen_count': face.seen_count,
            'name': face.name,
        }
        return json.dumps(data).encode('utf-8')
    
    def _deserialize_face(self, data: bytes) -> TrackedFace:
        """Deserialize bytes to a TrackedFace."""
        obj = json.loads(data.decode('utf-8'))
        embedding_bytes = base64.b64decode(obj['embedding'])
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(obj['embedding_shape'])
        
        return TrackedFace(
            face_id=obj['face_id'],
            embedding=embedding,
            last_seen=obj['last_seen'],
            seen_count=obj['seen_count'],
            name=obj.get('name'),
        )
    
    def load_all(self) -> Dict[int, TrackedFace]:
        faces = {}
        all_faces = self.redis.hgetall(self._faces_key)
        
        for face_id_bytes, face_data in all_faces.items():
            face = self._deserialize_face(face_data)
            faces[face.face_id] = face
        
        if faces:
            print(f"[RedisStorage] Loaded {len(faces)} faces")
        
        return faces
    
    def save_face(self, face: TrackedFace) -> None:
        self.redis.hset(self._faces_key, str(face.face_id), self._serialize_face(face))
    
    def delete_face(self, face_id: int) -> None:
        self.redis.hdel(self._faces_key, str(face_id))
    
    def get_next_id(self) -> int:
        return int(self.redis.incr(self._counter_key))
    
    def clear_all(self) -> None:
        self.redis.delete(self._faces_key)
        self.redis.delete(self._counter_key)
        print("[RedisStorage] Cleared all faces")
    
    def close(self) -> None:
        self.redis.close()


class MongoDBStorage(StorageBackend):
    """MongoDB storage backend for scalable persistence."""
    
    def __init__(
        self, 
        mongo_url: str = "mongodb://localhost:27017",
        database: str = "uniface",
        collection: str = "faces"
    ):
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("MongoDB support requires: pip install pymongo")
        
        self.client = MongoClient(mongo_url)
        self.db = self.client[database]
        self.collection = self.db[collection]
        self.counters = self.db["counters"]
        
        # Ensure index on face_id
        self.collection.create_index("face_id", unique=True)
        
        print(f"[MongoDBStorage] Connected to {mongo_url}, database: {database}")
    
    def _serialize_embedding(self, embedding: np.ndarray) -> dict:
        """Serialize numpy embedding for MongoDB."""
        return {
            'data': base64.b64encode(embedding.tobytes()).decode('ascii'),
            'shape': list(embedding.shape),
            'dtype': str(embedding.dtype),
        }
    
    def _deserialize_embedding(self, data: dict) -> np.ndarray:
        """Deserialize embedding from MongoDB."""
        arr_bytes = base64.b64decode(data['data'])
        return np.frombuffer(arr_bytes, dtype=data['dtype']).reshape(data['shape'])
    
    def load_all(self) -> Dict[int, TrackedFace]:
        faces = {}
        
        for doc in self.collection.find():
            face = TrackedFace(
                face_id=doc['face_id'],
                embedding=self._deserialize_embedding(doc['embedding']),
                last_seen=doc.get('last_seen', time.time()),
                seen_count=doc.get('seen_count', 1),
                name=doc.get('name'),
            )
            faces[face.face_id] = face
        
        if faces:
            print(f"[MongoDBStorage] Loaded {len(faces)} faces")
        
        return faces
    
    def save_face(self, face: TrackedFace) -> None:
        doc = {
            'face_id': face.face_id,
            'embedding': self._serialize_embedding(face.embedding),
            'last_seen': face.last_seen,
            'seen_count': face.seen_count,
            'name': face.name,
        }
        self.collection.replace_one(
            {'face_id': face.face_id},
            doc,
            upsert=True
        )
    
    def delete_face(self, face_id: int) -> None:
        self.collection.delete_one({'face_id': face_id})
    
    def get_next_id(self) -> int:
        result = self.counters.find_one_and_update(
            {'_id': 'face_id'},
            {'$inc': {'value': 1}},
            upsert=True,
            return_document=True
        )
        return result['value']
    
    def clear_all(self) -> None:
        self.collection.delete_many({})
        self.counters.delete_one({'_id': 'face_id'})
        print("[MongoDBStorage] Cleared all faces")
    
    def close(self) -> None:
        self.client.close()


def create_storage(
    storage_type: str,
    db_path: Optional[str] = None,
    redis_url: Optional[str] = None,
    mongo_url: Optional[str] = None,
    mongo_database: str = "uniface",
) -> StorageBackend:
    """Factory function to create storage backend."""
    
    if storage_type == "memory":
        return MemoryStorage()
    
    elif storage_type == "file":
        path = db_path or "faces.npz"
        return FileStorage(path)
    
    elif storage_type == "redis":
        url = redis_url or "redis://localhost:6379"
        return RedisStorage(url)
    
    elif storage_type == "mongodb":
        url = mongo_url or "mongodb://localhost:27017"
        return MongoDBStorage(url, database=mongo_database)
    
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


# =============================================================================
# Face Tracker
# =============================================================================

class FaceTracker:
    """
    Manages persistent face identification across frames.
    
    Uses face embeddings to recognize returning individuals and assign
    consistent IDs even after they leave and return to the camera view.
    """
    
    def __init__(
        self,
        storage: StorageBackend,
        similarity_threshold: float = 0.4,
        auto_save: bool = True,
        save_interval: int = 10,  # Save every N new faces
    ):
        """
        Initialize the face tracker.
        
        Args:
            storage: Storage backend for persistence.
            similarity_threshold: Minimum cosine similarity to consider a match (0-1).
                                  Higher values are stricter.
            auto_save: Whether to automatically save faces to storage.
            save_interval: How often to save (every N updates) for file storage.
        """
        self.storage = storage
        self.similarity_threshold = similarity_threshold
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._update_count = 0
        
        # Load existing faces from storage
        self.known_faces: Dict[int, TrackedFace] = storage.load_all()
        
        print(f"FaceTracker initialized with {len(self.known_faces)} known faces")
    
    def find_or_create_id(self, embedding: np.ndarray) -> Tuple[int, float, bool]:
        """
        Find a matching face ID or create a new one.
        
        Args:
            embedding: Normalized face embedding vector.
            
        Returns:
            Tuple of (face_id, similarity_score, is_new_face)
        """
        best_match_id = None
        best_similarity = -1.0
        
        # Compare against all known faces
        for face_id, tracked_face in self.known_faces.items():
            similarity = compute_similarity(
                embedding, 
                tracked_face.embedding, 
                normalized=True
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = face_id
        
        # Check if we found a match above threshold
        if best_match_id is not None and best_similarity >= self.similarity_threshold:
            # Update the tracked face with running average embedding
            tracked = self.known_faces[best_match_id]
            tracked.last_seen = time.time()
            tracked.seen_count += 1
            
            # Update embedding with exponential moving average for stability
            alpha = 0.1  # Weight for new embedding
            tracked.embedding = (1 - alpha) * tracked.embedding + alpha * embedding
            # Re-normalize after averaging
            tracked.embedding = tracked.embedding / np.linalg.norm(tracked.embedding)
            
            # Periodically save updates
            self._update_count += 1
            if self.auto_save and self._update_count % (self.save_interval * 10) == 0:
                self.storage.save_face(tracked)
            
            return best_match_id, best_similarity, False
        
        # Create new face ID
        new_id = self.storage.get_next_id()
        
        new_face = TrackedFace(
            face_id=new_id,
            embedding=embedding.copy(),
        )
        self.known_faces[new_id] = new_face
        
        # Save new face immediately
        if self.auto_save:
            self.storage.save_face(new_face)
        
        return new_id, 1.0, True
    
    def save_all(self) -> None:
        """Force save all faces to storage."""
        for face in self.known_faces.values():
            self.storage.save_face(face)
        print(f"Saved {len(self.known_faces)} faces to storage")
    
    def clear(self) -> None:
        """Clear all known faces."""
        self.storage.clear_all()
        self.known_faces.clear()
        print("Cleared all known faces")
    
    def close(self) -> None:
        """Close the tracker and storage connection."""
        self.storage.close()
    
    def get_stats(self) -> str:
        """Get tracker statistics."""
        return f"Known faces: {len(self.known_faces)}"


def draw_face_with_id(
    image: np.ndarray,
    bbox: List[float],
    face_id: int,
    confidence: float,
    similarity: float,
    is_new: bool,
    landmarks: Optional[np.ndarray] = None,
) -> None:
    """
    Draw face detection with ID overlay.
    
    Args:
        image: Image to draw on.
        bbox: Bounding box [x1, y1, x2, y2].
        face_id: Assigned face ID.
        confidence: Detection confidence.
        similarity: Recognition similarity score.
        is_new: Whether this is a newly registered face.
        landmarks: Optional facial landmarks to draw.
    """
    bbox = np.array(bbox, dtype=np.int32)
    x1, y1, x2, y2 = bbox
    
    # Calculate adaptive thickness based on face size
    face_width = x2 - x1
    thickness = max(2, int(face_width / 100))
    font_scale = max(0.5, face_width / 200)
    
    # Color coding: green for recognized, blue for new
    box_color = (0, 255, 0) if not is_new else (255, 200, 0)
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
    
    # Prepare ID label
    id_text = f"ID: {face_id}"
    
    # Calculate text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Draw background rectangle for text
    padding = 5
    cv2.rectangle(
        image,
        (x1, y1 - text_height - 2 * padding),
        (x1 + text_width + 2 * padding, y1),
        box_color,
        -1,  # Filled
    )
    
    # Draw ID text
    cv2.putText(
        image,
        id_text,
        (x1 + padding, y1 - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),  # Black text on colored background
        thickness,
    )
    
    # Draw similarity/confidence info below the box
    info_text = f"Sim: {similarity:.2f}" if not is_new else "NEW"
    cv2.putText(
        image,
        info_text,
        (x1, y2 + int(20 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.7,
        box_color,
        max(1, thickness - 1),
    )
    
    # Draw landmarks if provided
    if landmarks is not None:
        landmark_colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
        landmarks = np.array(landmarks, dtype=np.int32)
        for i, point in enumerate(landmarks):
            color = landmark_colors[i % len(landmark_colors)]
            cv2.circle(image, tuple(point), thickness + 1, color, -1)


def get_detector(name: str):
    """Create face detector by name."""
    if name == "scrfd":
        return SCRFD()
    return RetinaFace()


def get_recognizer(name: str):
    """Create face recognizer by name."""
    if name == "mobileface":
        return MobileFace()
    elif name == "sphereface":
        return SphereFace()
    return ArcFace()


def main():
    parser = argparse.ArgumentParser(
        description="Persistent Face ID Tracking with Webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # In-memory only (no persistence)
  python run_face_tracking.py --storage memory
  
  # File-based persistence (default)
  python run_face_tracking.py --storage file --save-db faces.npz
  
  # Redis persistence (real-time, survives restarts)
  python run_face_tracking.py --storage redis --redis-url redis://localhost:6379
  
  # MongoDB persistence (scalable, survives restarts)  
  python run_face_tracking.py --storage mongodb --mongo-url mongodb://localhost:27017

Controls:
  q - Quit
  s - Force save all faces
  r - Reset (clear all known faces)
  d - Toggle debug info
        """
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--detector", type=str, default="retinaface",
        choices=["retinaface", "scrfd"],
        help="Face detector model (default: retinaface)"
    )
    parser.add_argument(
        "--recognizer", type=str, default="arcface",
        choices=["arcface", "mobileface", "sphereface"],
        help="Face recognizer model (default: arcface)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Similarity threshold for face matching (0-1, default: 0.4)"
    )
    parser.add_argument(
        "--detection-threshold", type=float, default=0.6,
        help="Detection confidence threshold (default: 0.6)"
    )
    # Storage options
    parser.add_argument(
        "--storage", type=str, default="file",
        choices=["memory", "file", "redis", "mongodb"],
        help="Storage backend for face database (default: file)"
    )
    parser.add_argument(
        "--save-db", type=str, default="faces.npz",
        help="Path for file storage (default: faces.npz)"
    )
    parser.add_argument(
        "--redis-url", type=str, default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)"
    )
    parser.add_argument(
        "--mongo-url", type=str, default="mongodb://localhost:27017",
        help="MongoDB connection URL (default: mongodb://localhost:27017)"
    )
    parser.add_argument(
        "--mongo-database", type=str, default="uniface",
        help="MongoDB database name (default: uniface)"
    )
    parser.add_argument(
        "--show-landmarks", action="store_true",
        help="Show facial landmarks"
    )
    args = parser.parse_args()
    
    # Initialize storage backend
    print(f"Initializing storage: {args.storage}")
    try:
        storage = create_storage(
            storage_type=args.storage,
            db_path=args.save_db,
            redis_url=args.redis_url,
            mongo_url=args.mongo_url,
            mongo_database=args.mongo_database,
        )
    except (ImportError, ConnectionError) as e:
        print(f"Error: {e}")
        return
    
    # Initialize models
    print(f"Initializing detector: {args.detector}")
    detector = get_detector(args.detector)
    
    print(f"Initializing recognizer: {args.recognizer}")
    recognizer = get_recognizer(args.recognizer)
    
    # Initialize tracker with storage
    tracker = FaceTracker(
        storage=storage,
        similarity_threshold=args.threshold,
        auto_save=True,
    )
    
    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {args.camera}")
        tracker.close()
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Storage: {args.storage}")
    print("\nControls: [q]uit, [s]ave all, [r]eset, [d]ebug toggle")
    print("-" * 50)
    
    show_debug = True
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Process each detected face
            for face in faces:
                bbox = face["bbox"]
                confidence = face["confidence"]
                landmarks = np.array(face["landmarks"])
                
                # Skip low confidence detections
                if confidence < args.detection_threshold:
                    continue
                
                # Get face embedding
                embedding = recognizer.get_normalized_embedding(frame, landmarks)
                
                # Find or create face ID
                face_id, similarity, is_new = tracker.find_or_create_id(embedding)
                
                # Draw face with ID
                draw_face_with_id(
                    frame,
                    bbox,
                    face_id,
                    confidence,
                    similarity,
                    is_new,
                    landmarks if args.show_landmarks else None,
                )
            
            # Draw overlay info
            if show_debug:
                # FPS counter
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                # Face count
                cv2.putText(
                    frame,
                    f"Faces in frame: {len(faces)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                # Tracker stats
                cv2.putText(
                    frame,
                    f"{tracker.get_stats()} | Storage: {args.storage}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            
            # Display frame
            cv2.imshow("Face ID Tracking - Press 'q' to quit", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                tracker.save_all()
            elif key == ord('r'):
                tracker.clear()
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"Debug info: {'ON' if show_debug else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        
        print(f"\nSession summary:")
        print(f"  Total unique faces seen: {len(tracker.known_faces)}")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Storage: {args.storage}")


if __name__ == "__main__":
    main()
