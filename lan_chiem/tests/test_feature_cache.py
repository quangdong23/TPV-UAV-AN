import unittest
from types import SimpleNamespace

import torch
from lan_chiem.feature_cache import ImageFeatureCache


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixels, mask):
        self.calls += 1
        return [(pixels * 2, mask)], [pixels + 1]


class FeatureCacheTests(unittest.TestCase):
    def test_identical_prompt_batch_reuses_single_image_backbone(self):
        backbone = Backbone()
        model = SimpleNamespace(model=SimpleNamespace(backbone=backbone), training=False)
        cache = ImageFeatureCache()
        pixels, mask = torch.ones(1, 3, 4, 4), torch.ones(1, 4, 4)
        with torch.inference_mode(), cache.use(model):
            backbone(pixels, mask)
            features, positions = backbone(pixels.repeat(5, 1, 1, 1), mask.repeat(5, 1, 1))
            self.assertEqual(features[0][0].shape[0], 5)
            self.assertTrue(torch.equal(features[0][0], pixels.repeat(5, 1, 1, 1) * 2))
            positions.append(torch.zeros(1))
            self.assertEqual(len(backbone(pixels, mask)[1]), 1)
            different = pixels.repeat(2, 1, 1, 1)
            different[1] *= 3
            features, _ = backbone(different, mask.repeat(2, 1, 1))
            self.assertTrue(torch.equal(features[0][0], different * 2))
        self.assertEqual(backbone.calls, 2)

    def test_exact_inputs_reuse_features_but_changed_pixels_or_mask_miss(self):
        backbone = Backbone()
        model = SimpleNamespace(model=SimpleNamespace(backbone=backbone), training=False)
        cache = ImageFeatureCache()
        pixels, mask = torch.ones(1, 3, 4, 4), torch.ones(1, 4, 4)
        with torch.inference_mode(), cache.use(model):
            expected = backbone(pixels, mask)
            expected[1].append(torch.zeros(1))
            reused = backbone(pixels.clone(), mask.clone())
            self.assertEqual(len(reused[1]), 1)
            self.assertTrue(torch.equal(expected[0][0][0], reused[0][0][0]))
            backbone(pixels + 1, mask)
            backbone(pixels + 1, mask * 0)
        self.assertEqual(backbone.calls, 3)
        self.assertEqual(cache.hits, 1)
        self.assertNotIn('forward', backbone.__dict__)
        cache.clear()
        self.assertIsNone(cache.entry)

    def test_restores_forward_when_model_raises(self):
        backbone = Backbone()
        model = SimpleNamespace(model=SimpleNamespace(backbone=backbone), training=False)
        with self.assertRaises(RuntimeError):
            with ImageFeatureCache().use(model):
                raise RuntimeError('model failed')
        self.assertNotIn('forward', backbone.__dict__)

    def test_training_bypasses_cache_and_previous_override_is_restored(self):
        backbone = Backbone()
        model = SimpleNamespace(model=SimpleNamespace(backbone=backbone), training=True)
        prior = backbone.forward
        backbone.forward = prior
        cache = ImageFeatureCache()
        pixels, mask = torch.ones(1, 3, 2, 2), torch.ones(1, 2, 2)
        with cache.use(model):
            backbone(pixels, mask)
            backbone(pixels, mask)
        self.assertEqual(backbone.calls, 2)
        self.assertEqual(cache.hits, 0)
        model.training = False
        with torch.inference_mode(), cache.use(model):
            backbone(pixels, mask)
            backbone(pixels, mask)
        self.assertIs(backbone.forward, prior)
