import torch
import torch.nn as nn
import torch.fx as fx

from fxfusion.passes.fusion_pass import FusionPass
from fxfusion.passes.shape_prop import ShapePropPass
from fxfusion.passes.memory_plan import MemoryPlanningPass, TensorAlloc


def get_plan(model: nn.Module, x: torch.Tensor):
    fused = FusionPass().run(model)
    ShapePropPass(fused).propagate(x)
    planner = MemoryPlanningPass(fused)
    plan = planner.run()
    return fused, plan


def allocs_of_kind(fused: fx.GraphModule, kind: str) -> list[TensorAlloc]:
    return [
        node.meta["alloc"]
        for node in fused.graph.nodes
        if node.meta.get("kind") == kind
    ]


# -----------------------------------------------------------------------------
# Arena
# -----------------------------------------------------------------------------

def test_arena_size_nonzero():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
        def forward(self, x): return self.conv(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 3, 32, 32))
    assert plan.arena_size > 0


def test_arena_size_not_larger_than_naive():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
        def forward(self, x): return self.layers(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 256))
    activations = allocs_of_kind(fused, "activation")
    naive_size = sum(a.size_bytes for a in activations)
    assert plan.arena_size <= naive_size


# -----------------------------------------------------------------------------
# Offsets
# -----------------------------------------------------------------------------

def test_activation_offsets_within_arena():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        def forward(self, x): return self.conv2(self.conv1(x))

    fused, plan = get_plan(M().eval(), torch.randn(1, 3, 32, 32))
    activations = allocs_of_kind(fused, "activation")
    for alloc in activations:
        assert alloc.mem_offset is not None
        assert alloc.mem_offset >= 0
        assert alloc.mem_offset + alloc.size_bytes <= plan.arena_size


def test_offsets_are_8_byte_aligned():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
        def forward(self, x): return self.conv(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 3, 32, 32))
    activations = allocs_of_kind(fused, "activation")
    for alloc in activations:
        assert alloc.mem_offset is not None
        assert alloc.mem_offset % 8 == 0, f"Offset {alloc.mem_offset} not aligned"


def test_no_overlap_between_live_activations():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        def forward(self, x):
            a = self.conv1(x)
            b = self.conv2(a)
            return self.conv3(b) + a

    fused, plan = get_plan(M().eval(), torch.randn(1, 3, 32, 32))
    nodes = list(fused.graph.nodes)

    last_use: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for inp in node.all_input_nodes:
            last_use[inp.name] = i

    intervals = []
    for i, node in enumerate(nodes):
        alloc = node.meta.get("alloc")
        if alloc is None or alloc.kind != "activation" or alloc.mem_offset is None:
            continue
        live_end = last_use.get(node.name, i)
        intervals.append((alloc.mem_offset, alloc.mem_offset + alloc.size_bytes, i, live_end, node.name))

    for i, (s1, e1, def1, end1, n1) in enumerate(intervals):
        for s2, e2, def2, end2, n2 in intervals[i + 1:]:
            simultaneously_live = def1 <= end2 and def2 <= end1
            memory_overlap = s1 < e2 and s2 < e1
            assert not (simultaneously_live and memory_overlap), \
                f"Memory overlap between {n1} and {n2}"


# -----------------------------------------------------------------------------
# Kinds
# -----------------------------------------------------------------------------

def test_input_has_no_offset():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)
        def forward(self, x): return self.fc(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 64))
    inputs = allocs_of_kind(fused, "input")
    assert len(inputs) > 0
    for alloc in inputs:
        assert alloc.mem_offset is None
        assert alloc.alias_of is None


def test_consts_have_no_offset():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 64)
        def forward(self, x): return self.fc(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 64))
    consts = allocs_of_kind(fused, "const")
    assert len(consts) > 0
    for alloc in consts:
        assert alloc.mem_offset is None
        assert alloc.alias_of is None


def test_flatten_alias_shares_offset_with_source():
    class M(nn.Module):
        def forward(self, x): return torch.flatten(x, 1)

    fused, plan = get_plan(M().eval(), torch.randn(1, 64, 7, 7))
    aliases = allocs_of_kind(fused, "alias")
    name_to_alloc = {
        node.name: node.meta["alloc"]
        for node in fused.graph.nodes
        if "alloc" in node.meta
    }
    assert len(aliases) > 0
    for alloc in aliases:
        assert alloc.alias_of is not None
        assert alloc.alias_of in name_to_alloc
        source = name_to_alloc[alloc.alias_of]
        assert alloc.mem_offset == source.mem_offset
        assert alloc.size_bytes == source.size_bytes


def test_output_aliases_last_tensor():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 10)
        def forward(self, x): return self.fc(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 64))
    outputs = [
        node for node in fused.graph.nodes
        if node.meta.get("kind") == "output"
    ]
    assert len(outputs) == 1
    out_alloc = outputs[0].meta["alloc"]
    assert out_alloc.alias_of is not None


# -----------------------------------------------------------------------------
# Buffer Reuse
# -----------------------------------------------------------------------------

def test_buffer_reuse_occurs():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
            )
        def forward(self, x): return self.layers(x)

    fused, plan = get_plan(M().eval(), torch.randn(1, 256))
    activations = allocs_of_kind(fused, "activation")
    offsets = [a.mem_offset for a in activations if a.mem_offset is not None]
    assert len(offsets) > len(set(offsets)), \
        "Expected buffer reuse but all offsets are unique"