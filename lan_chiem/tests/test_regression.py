"""Additional regressions exercising production methods, with mocked model boundary."""
import unittest
import json
from pathlib import Path
from lan_chiem.detector import Detection, GroundingDINODetector, filter_tree_trunk_geometry
from PIL import Image
from lan_chiem.tests.test_pipeline_evidence import detector, row


class RegressionTests(unittest.TestCase):
    def test_new_ids_three_and_five_are_car_and_chair_not_old_table_vendor(self):
        for cid, name in [(3, 'Xe ô tô'), (5, 'Ghế nhựa'), (7, 'Biển quảng cáo'),
                          (8, 'Cột điện'), (9, 'Thân cây')]:
            d = detector(lambda text: [row(.7, text.split('.')[0].strip())])
            result = d.predict(Image.new('RGB', (200, 200)), target_classes=[cid], enable_counter_checks=False)
            self.assertEqual(result['catalog_version'], 'objects-v3-10')
            self.assertEqual(len(result['object_detections']), 1)
            self.assertEqual(result['object_detections'][0]['class_id'], cid)
            self.assertEqual(result['object_detections'][0]['class_name_vi'], name)
            self.assertFalse(result['review_candidates'])

    def test_merged_vendor_bicycle_remains_class_six_with_unknown_sale(self):
        d = detector(lambda _: [row(.7, 'vendor bicycle carrying goods')])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[6], enable_counter_checks=False)
        self.assertEqual(result['review_candidates'][0]['class_id'], 6)
        self.assertEqual(result['review_candidates'][0]['evidence']['sale_context_status'], 'sale_context_unknown')

    def test_saved_real_inference_partitions_match_current_postprocessing(self):
        folder = Path(__file__).resolve().parents[1] / 'outputs/evaluation'
        paths = [p for p in folder.glob('*/*_result.json')
                 if 'single_prompt' not in p.name and not p.name.startswith('5_tree_')]
        if not paths:
            self.skipTest('Optional saved real inference artifacts not present')
        for path in paths:
            result = json.loads(path.read_text(encoding='utf-8'))
            if 'object_detections' not in result or result.get('catalog_version') != 'objects-v3-10':
                continue
            d = GroundingDINODetector.__new__(GroundingDINODetector)
            d.config = result['provenance']['config_snapshot']
            hypotheses = []
            for raw in result['object_detections'] + result['review_candidates']:
                values = raw.copy()
                initial = raw['status_history'][0]
                values.update(status=initial['to'], review_reason=initial['reason'] if initial['to'] == 'review' else None,
                              status_history=[initial], related_detection_ids=[], conflict_info=None,
                              evidence={k: v for k, v in raw['evidence'].items() if k != 'spatial_relations'})
                hypotheses.append(Detection(**values))
            if result['config']['enable_tree_filter']:
                filter_tree_trunk_geometry([v for v in hypotheses if v.class_id == 9],
                                          result['image_size']['width'], result['image_size']['height'])
            counters = result.get('counter_evidence', result.get('debug_trace', {}).get('counter_evidence', {}))
            predicted, review, _ = d.analyze_conflicts_and_partition(hypotheses, counters)
            self.assertEqual({v.id for v in predicted}, {v['id'] for v in result['object_detections']}, str(path))
            self.assertEqual({v.id for v in review}, {v['id'] for v in result['review_candidates']}, str(path))

    def test_containment_is_directional_and_does_not_invent_cart_top(self):
        d = detector(lambda text: [row(.6, 'table', (10, 10, 190, 190))]
                     if text.startswith('plastic table') else [row(.7, 'food cart', (30, 30, 60, 60))])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[4, 6], enable_counter_checks=False)
        self.assertFalse(result['conflict_reports'])
        self.assertEqual({r['class_id'] for r in result['object_detections']}, {4, 6})

    def test_ambiguous_counter_stays_ambiguous_through_predict(self):
        d = detector(lambda text: [row(.6, 'food cart')] if text.startswith('street food cart')
                     else [row(.7, 'custom gate car')])
        d.counter_prompts = {'groups': {'barrier_gate': {'prompt': 'custom gate'},
                                       'car': {'prompt': 'car'}}}
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[6], debug=True)
        self.assertEqual(len(result['review_candidates']), 1)
        self.assertEqual(result['conflict_reports'][0]['conflict_type'], 'ambiguous_counter_overlap')
        self.assertEqual(result['counter_evidence']['_ambiguous'][0]['status'], 'ambiguous')
        self.assertFalse(result['counter_evidence']['car'])

    def test_explicit_zero_threshold_preserves_cli_precedence(self):
        d = detector(lambda _: [row(.22, 'table')])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[4],
                           box_threshold_override=0., preset=.30, enable_counter_checks=False)
        self.assertEqual(len(result['object_detections']), 1)
        self.assertEqual(result['config']['queried_details'][0]['threshold_source'], 'cli_box_threshold')

    def test_table_car_and_materials_are_neutral_observations(self):
        d = detector(lambda text: [row(.5, 'table')] if text.startswith('plastic table')
                     else [row(.6, 'car'), row(.7, 'construction materials')])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[4])
        kinds = {r['conflict_type'] for r in result['conflict_reports']}
        self.assertEqual(kinds, {'table_car_overlap', 'table_materials_overlap'})
        self.assertEqual(len(result['review_candidates']), 1)
        self.assertTrue(all('related_bbox_xyxy' in r for r in result['conflict_reports']))

    def test_pole_and_tree_both_hypotheses_preserved(self):
        d = detector(lambda text: [row(.6, 'utility pole' if text.startswith('utility') else 'tree trunk')])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[8, 9], enable_counter_checks=False)
        self.assertEqual({r['class_id'] for r in result['review_candidates']}, {8, 9})
        self.assertTrue(all(r['status_history'][-1]['to'] == 'review' for r in result['review_candidates']))

    def test_raw_label_shortening_preserves_query_hypothesis(self):
        d = detector(lambda _: [row(.5, 'basket crate')])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[10], enable_counter_checks=False)
        goods = result['review_candidates'][0]
        self.assertEqual(goods['queried_class_id'], 10)
        self.assertEqual(goods['raw_label'], 'basket crate')
        self.assertEqual(goods['evidence']['content_status'], 'container_only')

    def test_debug_counter_and_config_provenance(self):
        d = detector(lambda _: [])
        result = d.predict(Image.new('RGB', (200, 200)), target_classes=[6], debug=True)
        self.assertEqual(len(result['provenance']['config_sha256']), 64)
        self.assertEqual(result['provenance']['config_snapshot'], d.config)
        self.assertIn('torch', result['provenance']['versions'])
        self.assertIn('counter_evidence', result['debug_trace'])
