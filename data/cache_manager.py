import os
import torch
import logging
from utils.logger import get_logger
from typing import Any


class CacheManager:
    """
    A modular caching manager that supports in-memory (RAM) and disk-based caching.
    """

    def __init__(
        self,
        backend: str = "ram",
        cache_dir: str = ".dataset_cache",
        logger: logging.Logger = None,
    ):
        backend = backend.lower()
        if backend not in ["ram", "disk"]:
            raise ValueError(f"Unsupported cache backend: {backend}")
        self.backend = backend
        self.cache_dir = cache_dir
        if self.backend == "disk":
            os.makedirs(self.cache_dir, exist_ok=True)
        # For RAM caching, maintain an internal dictionary.
        self._ram_cache = {} if self.backend == "ram" else None
        self.logger = logger or get_logger(name=self.__class__.__name__)

    def is_cached(self, index: int) -> bool:
        """
        Check if an item at the given index is already cached.
        """
        if self.backend == "ram":
            return index in self._ram_cache
        elif self.backend == "disk":
            cache_file = os.path.join(self.cache_dir, f"item_{index}.pt")
            return os.path.exists(cache_file)
        return False

    def load(self, index: int) -> Any:
        """
        Load a cached item by index.
        """
        try:
            if self.backend == "ram":
                return self._ram_cache.get(index)
            elif self.backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{index}.pt")
                if os.path.exists(cache_file):
                    return torch.load(cache_file)
                else:
                    self.logger.warning(f"Cache file does not exist: {cache_file}")
                    return None
        except Exception as e:
            self.logger.error(f"Error loading cache for index {index}: {e}")
            return None

    def save(self, index: int, item: Any) -> None:
        """
        Save an item to the cache.
        """
        try:
            if self.backend == "ram":
                self._ram_cache[index] = item
            elif self.backend == "disk":
                cache_file = os.path.join(self.cache_dir, f"item_{index}.pt")
                torch.save(item, cache_file)
        except Exception as e:
            self.logger.error(f"Error saving cache for index {index}: {e}")

    def clear(self) -> None:
        """
        Clear the entire cache.
        """
        try:
            if self.backend == "ram":
                self._ram_cache.clear()
            elif self.backend == "disk":
                # Remove all cache files in the cache directory.
                for filename in os.listdir(self.cache_dir):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
