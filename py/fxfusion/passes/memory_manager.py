from enum import Enum, auto
import torch.fx as fx
from math import prod
from math import inf as INFINITY
from typing import Callable, Tuple, List

class AllocStrategy(Enum):
    BUMP_ONLY = auto()
    FIRST_FIT = auto()
    BEST_FIT  = auto()

class MemoryManager:
    
    def __init__(self, alignment: int = 64, strategy: AllocStrategy = AllocStrategy.BEST_FIT):
        if alignment <= 0:
            raise ValueError("alignment must be positive")

        if alignment & (alignment - 1) != 0:
            raise ValueError("alignment must be a power of two")
        
        self.alignment = alignment
        self.alloc_fn: Callable[[int], int] = self._select_alloc_fn(strategy)
        self.free_blocks: List[Tuple[int, int]] = []
        self.arena_top: int = 0
        self.peak_arena_top: int = 0

    def _align(self, value: int) -> int:
        return ((value + self.alignment - 1) // self.alignment) * self.alignment
        
    def tensor_size_bytes(self, node: fx.Node) -> int:
        dtype = node.meta.get("dtype")
        shape = node.meta.get("shape")

        if dtype is None or shape is None:
            raise RuntimeError(f"dtype and shape missing for node {node.name}")

        num_elements = prod(shape)
        bytes_per_element = dtype.itemsize

        return self._align(num_elements * bytes_per_element)

    def _select_alloc_fn(self, strategy: AllocStrategy) -> Callable[[int], int]:
        if strategy == AllocStrategy.BUMP_ONLY:
            return self._bump_alloc
        if strategy == AllocStrategy.FIRST_FIT:
            return self.first_fit_impl
        if strategy == AllocStrategy.BEST_FIT:
            return self.best_fit_impl
        raise ValueError(f"Unknown strategy: {strategy}")
    
    def allocate(self, size: int) -> int:
        return self.alloc_fn(size)
    
    def first_fit_impl(self, size: int) -> int:
        for idx, (block_offset, block_size) in enumerate(self.free_blocks):
            aligned_offset = self._align(block_offset)

            usable = block_offset + block_size - aligned_offset
            if usable >= size:
                self.free_blocks.pop(idx)
                self._split_block(aligned_offset, size, block_offset, block_size)
                return aligned_offset

        return self._bump_alloc(size)
    
    def best_fit_impl(self, size: int) -> int:
        best_idx, best_size = -1, INFINITY
        
        for idx, (block_offset, block_size) in enumerate(self.free_blocks):
            aligned_offset = self._align(block_offset)

            usable = block_offset + block_size - aligned_offset
            if usable >= size and usable < best_size:
                best_idx, best_size  = idx, usable

        if best_idx != -1:
            block_offset, block_size = self.free_blocks.pop(best_idx)
            aligned_offset = self._align(block_offset)
            self._split_block(aligned_offset, size, block_offset, block_size)
            return aligned_offset

        return self._bump_alloc(size)
    
    def _split_block(self, aligned_offset: int, size: int, block_offset: int, block_size: int) -> None:
        leftover_front = aligned_offset - block_offset
        if leftover_front > 0:
            self.free_blocks.append((block_offset, leftover_front))

        leftover_back = (block_size - size) - leftover_front
        if leftover_back > 0:
            self.free_blocks.append((aligned_offset + size, leftover_back))
            
    def _bump_alloc(self, size: int) -> int:
        aligned_offset = self._align(self.arena_top)
        self.arena_top = aligned_offset + size
        self.peak_arena_top = max(self.peak_arena_top, self.arena_top)
        return aligned_offset
    
    def release(self, offset: int, size: int) -> None:
        if size <= 0:
            return

        self.free_blocks.append((offset, size))
        self._coalesce()

        if self.free_blocks:
            last_block_offset, last_block_size = self.free_blocks[-1]

            if last_block_offset + last_block_size == self.arena_top:
                self.free_blocks.pop()
                self.arena_top = last_block_offset

    def _coalesce(self) -> None:
        if not self.free_blocks:
            return

        self.free_blocks.sort()
        merged = []

        for offset, size in self.free_blocks:
            if not merged:
                merged.append((offset, size))
                continue

            prev_offset, prev_size = merged[-1]

            if prev_offset + prev_size == offset:
                merged[-1] = (prev_offset, prev_size + size)

            else:
                merged.append((offset, size))

        self.free_blocks[:] = merged
