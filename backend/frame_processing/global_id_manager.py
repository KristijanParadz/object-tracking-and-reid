import numpy as np
import random
from typing import Dict, Tuple
from dataclasses import dataclass
from frame_processing.config import Config

Color = Tuple[int, int, int]

ClassID = int

ObjectID = int


@dataclass
class GlobalTrackEntry:
    """
    GlobalTrackEntry holds an embedding and its associated color.
    """
    embedding: np.ndarray
    color: Color


class GlobalIDManager:
    """
    GlobalIDManager manages global IDs associated with embeddings and their classes.

    Attributes:
        global_tracks (Dict[ClassID, Dict[ObjectID, GlobalTrackEntry]]): 
            Mapping of class IDs to their corresponding global IDs and associated GlobalTrackEntry.
        global_id_to_class (Dict[ObjectID, ClassID]): Mapping from global IDs to class IDs.
        next_global_id (ObjectID): Counter for the next available global ID.
    """

    def __init__(self) -> None:
        self.global_tracks: Dict[ClassID,
                                 Dict[ObjectID, GlobalTrackEntry]] = {}
        self.global_id_to_class: Dict[ObjectID, ClassID] = {}
        self.next_global_id: ObjectID = 1

    def _generate_random_color(self) -> Color:
        """
        Generates a random color represented as an RGB tuple.

        Returns:
            Color: A tuple of three integers between 0 and 255.
        """
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    def match_or_create(self, embedding: np.ndarray, class_id: ClassID) -> ObjectID:
        """
        Matches an embedding to an existing global ID or creates a new one if no match is found.

        Args:
            embedding (np.ndarray): The input embedding to match.
            class_id (ClassID): The identifier for the class/category.

        Returns:
            ObjectID: The matched or newly created global ID.
        """
        if not isinstance(embedding, np.ndarray):
            raise TypeError("embedding must be a numpy array.")

        if class_id not in self.global_tracks:
            self.global_tracks[class_id] = {}

        best_g_id: ObjectID = -1
        best_sim: float = -1.0

        for g_id, track in self.global_tracks[class_id].items():
            sim: float = float(np.dot(embedding, track.embedding))
            if sim > best_sim:
                best_sim = sim
                best_g_id = g_id

        if best_g_id != -1 and best_sim >= Config.SIMILARITY_THRESHOLD:
            return best_g_id

        color: Color = self._generate_random_color()
        new_g_id: ObjectID = self.next_global_id
        self.next_global_id += 1

        self.global_tracks[class_id][new_g_id] = GlobalTrackEntry(
            embedding, color)
        self.global_id_to_class[new_g_id] = class_id

        return new_g_id

    def get_color(self, global_id: ObjectID) -> Color:
        """
        Retrieves the color associated with a given global ID.

        Args:
            global_id (ObjectID): The global ID whose color is to be retrieved.

        Returns:
            Color: The color associated with the global ID, or white (255, 255, 255)
                   if the global ID is not found.
        """
        if global_id not in self.global_id_to_class:
            return (255, 255, 255)

        cls_id: ClassID = self.global_id_to_class[global_id]
        if cls_id in self.global_tracks and global_id in self.global_tracks[cls_id]:
            return self.global_tracks[cls_id][global_id].color
        return (255, 255, 255)

    def update_embedding(self, global_id: ObjectID, new_embedding: np.ndarray, alpha: float = 0.7) -> None:
        """
        Updates the embedding for a given global ID by blending the existing embedding with a new one.

        Args:
            global_id (ObjectID): The global ID whose embedding is to be updated.
            new_embedding (np.ndarray): The new embedding to blend with the existing one.
            alpha (float): Weight for the old embedding in the blending process (default 0.7).
        """
        if not isinstance(new_embedding, np.ndarray):
            raise TypeError("new_embedding must be a numpy array.")

        if global_id not in self.global_id_to_class:
            return

        cls_id: ClassID = self.global_id_to_class[global_id]
        if cls_id not in self.global_tracks or global_id not in self.global_tracks[cls_id]:
            return

        track = self.global_tracks[cls_id][global_id]
        # Blend the embeddings using a weighted sum.
        blended = alpha * track.embedding + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm
        self.global_tracks[cls_id][global_id] = GlobalTrackEntry(
            blended, track.color)

    def reset(self) -> None:
        """
        Resets the global tracking state by clearing all stored tracks and resetting the global ID counter.
        """
        self.global_tracks.clear()
        self.global_id_to_class.clear()
        self.next_global_id = 1
