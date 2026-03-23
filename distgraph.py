"""DistGraph v1 executor: validates and instantiates torch distributions from DistGraph dicts."""

import importlib
from typing import Any, Literal, Union

import torch
import torch.nn.functional as F
from pydantic import BaseModel, field_validator, model_validator


class DistGraphError(Exception):
    """Error during DistGraph validation or execution."""


# ---- Pydantic schema ----


class ConstNode(BaseModel):
    id: str
    kind: Literal["const"]
    value: Any  # scalar, list, or nested lists

    @field_validator("value")
    @classmethod
    def value_must_be_numeric(cls, v):
        def _check(x):
            if isinstance(x, (int, float)):
                return
            if isinstance(x, list):
                for item in x:
                    _check(item)
                return
            raise ValueError(f"const value must be numeric, got {type(x).__name__}")

        _check(v)
        return v


class TensorOpNode(BaseModel):
    id: str
    kind: Literal["tensor_op"]
    op: str
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    @field_validator("op")
    @classmethod
    def op_must_be_whitelisted(cls, v):
        if v not in TENSOR_OPS:
            raise ValueError(f"unknown op {v!r}, allowed: {sorted(TENSOR_OPS)}")
        return v


class CallNode(BaseModel):
    id: str
    kind: Literal["call"]
    callable: str
    returns: Literal["distribution", "transform", "object"] = "distribution"
    kwargs: dict[str, Any] = {}
    label: str | None = None

    @field_validator("callable")
    @classmethod
    def callable_must_be_allowed(cls, v):
        if not any(v.startswith(p) for p in CALLABLE_PREFIXES):
            raise ValueError(
                f"callable {v!r} not in allowlist (must start with "
                f"{', '.join(CALLABLE_PREFIXES)})"
            )
        return v


Node = Union[ConstNode, TensorOpNode, CallNode]


def _parse_node(data: dict) -> Node:
    """Parse a node dict into the appropriate pydantic model."""
    kind = data.get("kind")
    if kind == "const":
        return ConstNode(**data)
    if kind == "tensor_op":
        return TensorOpNode(**data)
    if kind == "call":
        return CallNode(**data)
    raise DistGraphError(f"Unknown node kind: {kind!r}")


class DistGraph(BaseModel):
    format: Literal["mossaic.distgraph"]
    version: Literal[1]
    root: str
    nodes: list[dict[str, Any]]
    meta: dict[str, Any] | None = None

    @model_validator(mode="after")
    def root_must_exist(self):
        node_ids = set()
        for n in self.nodes:
            nid = n.get("id")
            if nid is None:
                raise ValueError(f"Node missing 'id': {n}")
            if nid in node_ids:
                raise ValueError(f"Duplicate node id: {nid!r}")
            node_ids.add(nid)
        if self.root not in node_ids:
            raise ValueError(f"root {self.root!r} not found in nodes")
        return self


# ---- Op whitelist and callable allowlist ----

TENSOR_OPS = {
    # arithmetic
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "neg": torch.neg,
    # linear algebra / shape
    "matmul": torch.matmul,
    "einsum": torch.einsum,
    "transpose": torch.transpose,
    "reshape": torch.reshape,
    "squeeze": torch.squeeze,
    "unsqueeze": torch.unsqueeze,
    "sum": torch.sum,
    "mean": torch.mean,
    "norm": torch.linalg.norm,
    "square": torch.square,
    "sqrt": torch.sqrt,
    "stack": torch.stack,
    "cat": torch.cat,
    "diag_embed": torch.diag_embed,
    # nonlinearities
    "exp": torch.exp,
    "log": torch.log,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": F.relu,
    "clamp": torch.clamp,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    # numerical
    "where": torch.where,
}

CALLABLE_PREFIXES = (
    "torch.distributions.",
    "pyro.distributions.",
)


# ---- Execution helpers ----


def _resolve_refs(value, env):
    """Recursively resolve {"$ref": "node_id"} placeholders."""
    if isinstance(value, dict):
        if "$ref" in value and len(value) == 1:
            ref_id = value["$ref"]
            if ref_id not in env:
                raise DistGraphError(f"Unresolved $ref: {ref_id!r}")
            return env[ref_id]
        return {k: _resolve_refs(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(v, env) for v in value]
    return value


def _import_callable(path):
    """Import a class/function from a dotted path, checking the allowlist."""
    if not any(path.startswith(p) for p in CALLABLE_PREFIXES):
        raise DistGraphError(f"Callable not in allowlist: {path!r}")
    module_path, _, attr_name = path.rpartition(".")
    if not module_path:
        raise DistGraphError(f"Invalid callable path: {path!r}")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise DistGraphError(f"Cannot import {path!r}: {e}") from e


# ---- Main API ----


def validate(graph):
    """Validate a DistGraph dict using pydantic schema. Raises DistGraphError."""
    try:
        dg = DistGraph(**graph)
        # Also validate each node individually
        for node_dict in dg.nodes:
            _parse_node(node_dict)
    except Exception as e:
        raise DistGraphError(str(e)) from e


def execute(graph, expected_event_shape=None):
    """Execute a DistGraph dict and return the root Distribution.

    Args:
        graph: DistGraph dict.
        expected_event_shape: if provided, assert dist.event_shape matches.

    Returns:
        torch.distributions.Distribution
    """
    validate(graph)

    env = {}
    for node_dict in graph["nodes"]:
        node = _parse_node(node_dict)
        nid = node.id

        try:
            if isinstance(node, ConstNode):
                # Always float64 — scoring samples are float64 and torch
                # distributions require matching dtypes across parameters.
                env[nid] = torch.tensor(node.value, dtype=torch.float64)

            elif isinstance(node, TensorOpNode):
                args = _resolve_refs(node.args, env)
                kwargs = _resolve_refs(node.kwargs, env)
                env[nid] = TENSOR_OPS[node.op](*args, **kwargs)

            elif isinstance(node, CallNode):
                cls = _import_callable(node.callable)
                kwargs = _resolve_refs(node.kwargs, env)
                env[nid] = cls(**kwargs)

        except DistGraphError:
            raise
        except Exception as e:
            raise DistGraphError(f"Node {nid!r} ({node.kind}): {e}") from e

    root = env[graph["root"]]
    if not isinstance(root, torch.distributions.Distribution):
        raise DistGraphError(f"Root node is not a Distribution: {type(root).__name__}")

    if expected_event_shape is not None:
        if root.event_shape != torch.Size(expected_event_shape):
            raise DistGraphError(
                f"event_shape mismatch: expected {expected_event_shape}, "
                f"got {tuple(root.event_shape)}"
            )

    return root
