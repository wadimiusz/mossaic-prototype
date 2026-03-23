# EVOLVE-BLOCK-START
def build_dist(model_weights, model_config):
    """Build a distribution q(y) over residual stream vectors of a transformer.

    model_weights: dict of numpy arrays — the transformer's weights
    model_config: dict — includes d_model (64), nhead, num_layers, dim_ff, seed,
        and possibly other task-specific architectural parameters.

    Returns: DistGraph v1 dict describing q(residual_stream_vector).
    """
    d = model_config["d_model"]
    return {
        "format": "mossaic.distgraph",
        "version": 1,
        "root": "y_dist",
        "nodes": [
            {"id": "mu", "kind": "const", "value": [0.0] * d},
            {"id": "raw_scale", "kind": "const", "value": [0.0] * d},
            {
                "id": "scale",
                "kind": "tensor_op",
                "op": "softplus",
                "args": [{"$ref": "raw_scale"}],
                "kwargs": {},
            },
            {
                "id": "base",
                "kind": "call",
                "callable": "torch.distributions.Normal",
                "returns": "distribution",
                "kwargs": {"loc": {"$ref": "mu"}, "scale": {"$ref": "scale"}},
            },
            {
                "id": "y_dist",
                "kind": "call",
                "callable": "torch.distributions.Independent",
                "returns": "distribution",
                "kwargs": {
                    "base_distribution": {"$ref": "base"},
                    "reinterpreted_batch_ndims": 1,
                },
            },
        ],
    }


# EVOLVE-BLOCK-END
