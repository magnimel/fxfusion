import torch.fx as fx
from math import prod
from typing import Tuple, List

class MemoryManager:
    
    def __init__(self, alignment: int = 64):
        if alignment <= 0:
            raise ValueError("alignment must be positive")

        if alignment & (alignment - 1) != 0:
            raise ValueError("alignment must be a power of two")

        self.alignment = alignment
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

    def allocate(self, size: int) -> int:
        for idx, (block_offset, block_size) in enumerate(self.free_blocks):
            aligned_offset = self._align(block_offset)

            if aligned_offset + size <= block_offset + block_size:
                self.free_blocks.pop(idx)

                leftover_front = aligned_offset - block_offset
                if leftover_front > 0:
                    self.free_blocks.append((block_offset, leftover_front))

                leftover_back = (block_size - size) - leftover_front
                if leftover_back > 0:
                    self.free_blocks.append((aligned_offset + size, leftover_back))

                return aligned_offset

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

    def _coalesce(self, DEBUG: bool = False) -> None:
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
