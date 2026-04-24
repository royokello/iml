from .layer_cache import Flux2KVLayerCache


class Flux2KVCache:
    """Container for all layers' reference-token KV caches.

    Holds separate cache lists for double-stream and single-stream transformer blocks.
    """

    def __init__(self, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [Flux2KVLayerCache() for _ in range(num_double_layers)]
        self.single_block_caches = [Flux2KVLayerCache() for _ in range(num_single_layers)]
        self.num_ref_tokens: int = 0

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def clear(self):
        for cache in self.double_block_caches:
            cache.clear()
        for cache in self.single_block_caches:
            cache.clear()
        self.num_ref_tokens = 0
