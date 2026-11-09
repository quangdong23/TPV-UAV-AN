"""Exercise production predict(), crop inference and counter checks without weights."""
import copy
import json
import unittest
import tempfile
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from lan_chiem.detector import Detection, GroundingDINODetector, merge_duplicate_carts
from lan_chiem.visualization import draw_detections

ROOT = Path(__file__).resolve().parents[1]


class Processor:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.current = ""

    def __call__(self, images, text, return_tensors):
        self.current = text
        self.calls.append(text)
        return {"input_ids": torch.tensor([[1]])}

    def post_process_grounded_object_detection(self, **kwargs):
        rows = self.responses(self.current)
        rows = [r for r in rows if r[0] >= kwargs["threshold"]]
        return [{"scores": torch.tensor([r[0] for r in rows]),
                 "boxes": torch.tensor([r[1] for r in rows]).reshape(-1, 4),
                 "text_labels": [r[2] for r in rows]}]


def detector(responses):
    obj = GroundingDINODetector(cache_dir=ROOT / 'outputs' / 'test_cache')
    obj.config['inference']['query_mode'] = 'per_class'
    obj.config['inference']['counter_policy'] = 'all'
    obj.config['review_rules']['motorcycle_bicycle_overlap'] = True
    obj.processor = Processor(responses)
    obj.model = lambda **kwargs: object()
    return obj


def row(score=.6, label='person', box=(10, 10, 80, 80)):
    return (score, list(box), label)


class PipelineEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (200, 200))

    def test_debug_invariance_and_real_class_threshold(self):
        d = detector(lambda _: [row(.22, 'table'), row(.18, 'table', (100, 100, 140, 140))])
        a = d.predict(self.image, target_classes=[4], enable_counter_checks=False)
        b = d.predict(self.image, target_classes=[4], debug=True, enable_counter_checks=False)
        self.assertEqual(a['object_detections'], b['object_detections'])
        self.assertEqual(len(a['object_detections']), 1)
        self.assertEqual(b['debug_trace']['total_candidates_collected'], 2)
        self.assertEqual(a['config']['queried_details'][0]['threshold_source'], 'class_config')
        c = d.predict(self.image, target_classes=[4], preset=.25, enable_counter_checks=False)
        self.assertEqual(c['object_detections'], [])
        self.assertEqual(c['config']['queried_details'][0]['threshold_source'], 'cli_preset')

    def test_class_isolation_sends_only_class_six_prompt(self):
        d = detector(lambda _: [])
        d.predict(self.image, target_classes=[6], enable_counter_checks=False)
        self.assertEqual(d.processor.calls, [d.classes[6]['prompt'] + ' .'])
        with self.assertRaises(ValueError):
            d.predict(self.image, target_classes=[11])
        self.assertEqual(len(d.processor.calls), 1)

    def test_grouped_ambiguous_cannot_suppress_clear_or_be_promoted(self):
        d = detector(lambda _: [row(.9, 'person motorcycle'), row(.5, 'person')])
        result = d.predict(self.image, query_mode='grouped', target_groups=['people_vehicles'],
                           enable_counter_checks=False)
        self.assertEqual(len(result['object_detections']), 1)
        self.assertEqual(result['object_detections'][0]['class_id'], 1)
        ambiguous = result['review_candidates'][0]
        self.assertIsNone(ambiguous['class_id'])
        self.assertEqual(ambiguous['candidate_class_ids'], [1, 2])
        self.assertTrue(all(h['to'] == 'review' for h in ambiguous['status_history']))
        again = d.analyze_conflicts_and_partition(result['_review_objects'])
        self.assertFalse(again[0])
        json.dumps({k: v for k, v in result.items() if not k.startswith('_')})

    def test_custom_counter_is_sent_and_ambiguous_counter_preserved(self):
        d = detector(lambda _: [row(.6, 'custom gate bicycle')])
        d.counter_prompts = {'groups': {
            'barrier_gate': {'prompt': 'custom gate', 'aliases': ['custom gate']},
            'bicycle': {'prompt': 'bicycle', 'aliases': ['bicycle']}}}
        result = d.run_counter_checks(self.image)
        self.assertEqual(d.processor.calls, ['custom gate . bicycle .'])
        self.assertFalse(result['barrier_gate'])
        self.assertFalse(result['bicycle'])
        self.assertEqual(result['_ambiguous'][0]['candidate_counter_groups'], ['barrier_gate', 'bicycle'])
        self.assertIn('query_prompt', result['_ambiguous'][0])

    def test_goods_containers_and_fruit_retained_without_sale_verification(self):
        d = detector(lambda text: [row(.5, 'basket crate'), row(.55, 'fruit', (110, 110, 150, 150))]
                     if text.startswith('basket') else [row(.7, 'table')])
        result = d.predict(self.image, target_classes=[4, 10], enable_counter_checks=False)
        goods = [r for r in result['review_candidates'] if r['class_id'] == 10]
        self.assertEqual(len(goods), 2)
        self.assertEqual({r['evidence']['content_status'] for r in goods}, {'container_only', 'content_candidate'})
        self.assertEqual(result['summary']['verified_goods_count'], 0)
        self.assertTrue(any(r['conflict_type'] == 'overlap_with_table' for r in result['conflict_reports']))
        self.assertNotIn('dining_food_as_goods', str(result['conflict_reports']))
        d.config['review_rules']['overlap_with_table'] = False
        off = d.predict(self.image, target_classes=[4, 10], enable_counter_checks=False)
        self.assertFalse(off['conflict_reports'])

    def test_bicycle_and_other_conflicts_use_actual_pipeline(self):
        d = detector(lambda text: [row(.6, 'motorcycle')] if text.startswith('motorcycle')
                     else [row(.65, 'bicycle')])
        result = d.predict(self.image, target_classes=[2], debug=True)
        self.assertTrue(any(r['conflict_type'] == 'motorcycle_bicycle_overlap' for r in result['conflict_reports']))
        self.assertEqual(result['review_candidates'][0]['class_id'], 2)
        self.assertIn('counter_evidence', result['debug_trace'])

    def test_tile_production_translates_once_and_small_tiled_runs(self):
        d = detector(lambda _: [row(.6, 'person', (10, 20, 50, 60))])
        result = d.predict(self.image, mode='tiled', tile_size=120, overlap_ratio=0,
                           target_classes=[1], enable_counter_checks=False)
        boxes = [r['bbox_xyxy'] for r in result['object_detections']]
        self.assertIn([90., 100., 130., 140.], boxes)
        small = d.predict(Image.new('RGB', (80, 80)), mode='tiled', tile_size=120,
                          target_classes=[1], enable_counter_checks=False)
        self.assertEqual(len(small['object_detections']), 1)

    def test_real_saved_cart_boxes_and_adjacent_carts(self):
        target = ROOT / 'outputs/4_result.json'
        if not target.exists():
            self.skipTest('outputs/4_result.json not present')
        saved = json.loads(target.read_text(encoding='utf-8'))
        debug = json.loads((ROOT / 'outputs/4_debug.json').read_text(encoding='utf-8'))
        base = next(v for v in saved['object_detections'] if v['id'] == 'det_11')
        duplicates = [r for r in debug['rejected_records']
                      if r['reason'] == 'suppressed_duplicate_cart_heuristic' and 'det_11' in r['details']]
        self.assertTrue(duplicates)
        def cart(id, box, score):
            return Detection(id, 6, 'cart', 'Cart', 'food cart', 'class_6', 'full', score, box)
        carts = [cart('real', base['bbox_xyxy'], base['score'])]
        carts += [cart(f'dup{i}', r['bbox_xyxy'], r['score']) for i, r in enumerate(duplicates)]
        self.assertEqual(len(merge_duplicate_carts(carts)), 1)
        adjacent = copy.deepcopy(carts[0])
        adjacent.id = 'adjacent'
        adjacent.bbox_xyxy = [v + 200 if i % 2 == 0 else v for i, v in enumerate(adjacent.bbox_xyxy)]
        self.assertEqual(len(merge_duplicate_carts([carts[0], adjacent])), 2)

    def test_tree_scope_keeps_both_saved_hypotheses(self):
        boxes = [(839.29, 111.44, 901.96, 443.84), (694.23, 1.02, 1216.72, 446.02)]
        d = detector(lambda _: [row(.5, 'tree trunk tree stem', box) for box in boxes])
        result = d.predict(Image.new('RGB', (1804, 1004)), target_classes=[9], enable_counter_checks=False)
        self.assertEqual(len(result['review_candidates']), 2)
        self.assertTrue(all('tree_scope_uncertain' in r['review_reason'] for r in result['review_candidates']))

    def test_single_prompt_renders_raw_label_without_fake_class(self):
        d = detector(lambda _: [row(.5, 'food vending cart')])
        result = d.predict(self.image, single_prompt='food vending cart')
        from lan_chiem import visualization
        original = visualization._draw_single_box_and_badge
        with patch.object(visualization, '_draw_single_box_and_badge', wraps=original) as draw:
            image = draw_detections(self.image, result['detections'])
            self.assertIsNone(draw.call_args.kwargs['cid'])
            self.assertEqual(draw.call_args.kwargs['raw_label'], 'food vending cart')
            self.assertIsInstance(image, Image.Image)


    def test_cli_writer_serializes_arguments_and_cleans_stale_debug(self):
        from lan_chiem.predict import parse_args, process_single_image
        d = detector(lambda _: [row(.5, 'person')])
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            out = Path(folder)
            path = out / 'input.png'
            self.image.save(path)
            with patch('sys.argv', ['predict', '--input', str(path), '--output', str(out), '--class-id', '1']):
                args = parse_args()
                (out / 'input_debug.json').write_text('{}')
                process_single_image(path, d, out, args, [1], None, None)
            result = json.loads((out / 'input_result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['provenance']['cli_arguments']['class_id'], 1)
            self.assertFalse((out / 'input_debug.json').exists())
            self.assertTrue((out / 'input_annotated.jpg').exists())

    def test_real_saved_six_cart_duplicates(self):
        target = ROOT / 'outputs/6_result.json'
        if not target.exists():
            self.skipTest('outputs/6_result.json not present')
        result = json.loads(target.read_text(encoding='utf-8'))
        debug = json.loads((ROOT / 'outputs/6_debug.json').read_text(encoding='utf-8'))
        kept = next(d for d in result['object_detections'] if d['id'] == 'det_36')
        rejected = [r for r in debug['rejected_records'] if r['reason'] == 'suppressed_duplicate_cart_heuristic'
                    and 'det_36' in r['details']]
        self.assertTrue(rejected)
        def cart(id, box, score):
            return Detection(id, 6, 'cart', 'Cart', 'cart', 'class_6', 'full', score, box)
        hypotheses = [cart('saved', kept['bbox_xyxy'], kept['score'])]
        hypotheses += [cart(str(i), r['bbox_xyxy'], r['score']) for i, r in enumerate(rejected)]
        self.assertEqual(len(merge_duplicate_carts(hypotheses)), 1)

    def test_cli_raw_summary_does_not_claim_zero_predictions(self):
        from lan_chiem.predict import main
        d = detector(lambda _: [row(.5, 'food cart')])
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as folder:
            image = Path(folder) / 'sample.png'
            self.image.save(image)
            stream = io.StringIO()
            with patch('sys.argv', ['predict', '--backend', 'grounding-dino', '--input', str(image), '--output', folder, '--prompt', 'food cart']), \
                 patch('lan_chiem.detector.GroundingDINODetector', return_value=d), redirect_stdout(stream):
                self.assertEqual(main(), 0)
            self.assertIn('Raw predictions (chưa xác minh): 1', stream.getvalue())
            self.assertNotIn('Không có đối tượng', stream.getvalue())

    def test_tiny_review_box_can_be_rendered(self):
        from lan_chiem.visualization import draw_candidates_overlay
        candidate = {'class_id': 10, 'bbox_xyxy': [10, 10, 13, 13], 'score': .3}
        self.assertIsInstance(draw_candidates_overlay(self.image, [], [candidate]), Image.Image)


if __name__ == '__main__':
    unittest.main()
