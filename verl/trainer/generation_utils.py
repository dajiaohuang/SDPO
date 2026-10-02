import numpy as np


def chats_to_object_array(chats: list) -> np.ndarray:
    """Store each chat as one array element, even when turn counts differ."""
    result = np.empty(len(chats), dtype=object)
    result[:] = chats
    return result
