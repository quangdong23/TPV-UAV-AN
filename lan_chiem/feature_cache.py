"""Bounded, exact-input cache for Grounding DINO's text-independent backbone.

Only the most recent image/crop is retained. Encoder outputs must never be
cached across captions because Grounding DINO fuses text inside the encoder.
"""
from contextlib import contextmanager

import torch


class ImageFeatureCache:
    def __init__(self):
        self.clear()

    def clear(self):
        self.entry = None
        self.hits = 0
        self.misses = 0

    @contextmanager
    def use(self, model):
        backbone = getattr(getattr(model, "model", None), "backbone", None)
        if backbone is None or model.training:
            yield
            return
        original = backbone.forward
        had_override = "forward" in backbone.__dict__
        prior_override = backbone.__dict__.get("forward")

        def forward(pixel_values, pixel_mask):
            batch_size = pixel_values.shape[0]
            shared = (batch_size > 1
                      and torch.equal(pixel_values, pixel_values[:1].expand_as(pixel_values))
                      and torch.equal(pixel_mask, pixel_mask[:1].expand_as(pixel_mask)))
            # Different captions for the same image share the text-independent
            # backbone. Never share text-fused encoder outputs.
            base_pixels = pixel_values[:1] if shared else pixel_values
            base_mask = pixel_mask[:1] if shared else pixel_mask
            previous = self.entry
            if (previous is not None and previous[0] is backbone
                    and self._equal(previous[1], base_pixels)
                    and self._equal(previous[2], base_mask)):
                self.hits += 1
                features, positions = previous[3]
            else:
                features, positions = original(base_pixels, base_mask)
                self.entry = (backbone, base_pixels.clone(), base_mask.clone(),
                              (features, list(positions)))
                self.misses += 1
            # Transformers appends extra feature-level positions to this list.
            if shared:
                expand = lambda tensor: tensor.expand(batch_size, *tensor.shape[1:])
                return [(expand(feature), expand(mask)) for feature, mask in features], [expand(p) for p in positions]
            return list(features), list(positions)

        backbone.forward = forward
        try:
            yield
        finally:
            if had_override:
                backbone.forward = prior_override
            else:
                del backbone.forward

    @staticmethod
    def _equal(a, b):
        return (a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                and torch.equal(a, b))
