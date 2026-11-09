"""Pass counts and conservative dedup on production paths and saved real boxes."""
import unittest
from pathlib import Path
from PIL import Image
from lan_chiem.detector import Detection
from lan_chiem.postprocess import consolidate_nested_boxes
from lan_chiem.tests.test_pipeline_evidence import detector, row

ROOT = Path(__file__).resolve().parents[1]


def detection(id, cid, box, score=.6):
    return Detection(id, cid, str(cid), str(cid), 'raw', 'test', 'full', score, box)


class CompactTests(unittest.TestCase):
    def test_configured_main_passes_and_no_unneeded_counter(self):
        d = detector(lambda text: [row(.7, 'person')] if text.startswith('person') else [])
        d.config['inference']['counter_policy'] = 'selective'
        d.config['review_rules']['motorcycle_bicycle_overlap'] = False
        d.config['inference']['compact_group_keys'] = ['compact_objects', 'compact_trunks']
        d.prompt_groups['compact_objects']['class_ids'] = [1, 2, 3, 4, 5, 6]
        result = d.predict(Image.new('RGB', (200, 200)), query_mode='compact')
        self.assertEqual(result['model_calls'], len(d.config['inference']['compact_group_keys']))
        self.assertEqual(result['counter_plan']['queried_groups'], [])
        self.assertEqual(len(result['config']['queried_details']), len(d.config['inference']['compact_group_keys']))

    def test_car_reused_and_bicycle_not_requested(self):
        d = detector(lambda text: [row(.7, 'car'), row(.6, 'street food cart')]
                     if text.startswith('person') else [])
        d.config['inference']['counter_policy'] = 'selective'
        d.config['review_rules']['motorcycle_bicycle_overlap'] = False
        d.config['inference']['compact_group_keys'] = ['compact_objects', 'compact_trunks']
        d.prompt_groups['compact_objects']['class_ids'] = [1, 2, 3, 4, 5, 6]
        result = d.predict(Image.new('RGB', (200, 200)), query_mode='compact')
        self.assertEqual(result['model_calls'], 2)
        counter_prompt = d.processor.calls[-1]
        self.assertNotIn('automobile', counter_prompt)
        self.assertNotIn('bicycle', counter_prompt)
        self.assertEqual(result['counter_evidence']['car'][0]['origin'], 'main_detection')
        self.assertEqual({r['class_id'] for r in result['review_candidates']}, {3,6})
        self.assertFalse(result['object_detections'])

    def test_saved_nested_pairs_merge_but_different_objects_do_not(self):
        # Representative nested scopes from the observed table/pole/chair/sign failures.
        # Fixtures live here so deleting user outputs cannot break the suite.
        for cid, inner in [(4,[0,25.05,100,74.95]), (8,[28,0,72,100]),
                           (5,[0,32.05,100,67.95]), (7,[0,32.35,100,67.65])]:
            boxes = [detection('outer',cid,[0,0,100,100],.7),
                     detection('inner',cid,inner,.6)]
            kept, removed = consolidate_nested_boxes(boxes)
            self.assertEqual(len(kept),1, cid)
            self.assertEqual(len(removed),1)
            self.assertEqual(len(kept[0].evidence['duplicate_hypotheses']),1)
        a = detection('a', 5, [0,0,30,30])
        b = detection('b', 5, [35,0,65,30])
        self.assertEqual(len(consolidate_nested_boxes([a,b])[0]), 2)

    def test_aggregate_box_does_not_swallow_cars(self):
        boxes = [detection('row',3,[0,0,100,200],.9),
                 detection('car1',3,[0,0,100,80]), detection('car2',3,[0,100,100,180])]
        kept, removed = consolidate_nested_boxes(boxes)
        self.assertEqual(len(kept),3)
        self.assertFalse(removed)
        self.assertEqual(boxes[0].status,'review')
        self.assertEqual(boxes[0].evidence['contained_detection_ids'],['car1','car2'])

    def test_trunk_scopes_and_two_riders_are_not_auto_merged(self):
        boxes = [detection('trunk1',9,[0,0,40,100]), detection('trunk2',9,[0,0,80,120])]
        self.assertEqual(len(consolidate_nested_boxes(boxes)[0]),2)
        riders = [detection('rider1',1,[1471.82,719.34,1644.59,904.19]),
                  detection('rider2',1,[1499.6,791.61,1682.83,996.42])]
        self.assertEqual(len(consolidate_nested_boxes(riders)[0]),2)

    def test_non_target_and_mixed_phrase_are_not_promoted(self):
        d = detector(lambda text: [row(.7,'bicycle'), row(.6,'motorcycle bicycle',(100,100,170,170))])
        d.config['inference']['compact_group_keys'] = ['contrast_test']
        d.config['inference']['counter_policy'] = 'selective'
        d.prompt_groups['contrast_test'] = {'class_ids':[2], 'box_threshold':.25,
            'distractors':{'bicycle':{'prompt':'bicycle','aliases':['bicycle']}}}
        r = d.predict(Image.new('RGB',(200,200)), query_mode='compact', debug=True)
        self.assertFalse(r['object_detections'])
        self.assertEqual(len(r['review_candidates']),1)
        self.assertIn('target_and_non_target_phrase_unresolved',r['review_candidates'][0]['review_reason'])
        self.assertEqual(r['model_calls'],1)
        self.assertFalse(r['counter_plan']['queried_groups'])
        self.assertEqual(len(r['counter_evidence']['bicycle']),2)

    def test_same_caption_partial_overlap_does_not_downgrade_motorcycle(self):
        d = detector(lambda _: [])
        d.config['review_rules']['motorcycle_bicycle_overlap'] = True
        m = detection('motorcycle',2,[0,0,100,100])
        partial = {'id':'bicycle','bbox_xyxy':[0,0,30,100], 'score':.8,'origin':'same_caption'}
        predicted, review, _ = d.analyze_conflicts_and_partition([m],{'bicycle':[partial]})
        self.assertEqual(len(predicted),1)
        self.assertFalse(review)
        other = dict(partial,bbox_xyxy=[1,1,99,99])
        _, review, _ = d.analyze_conflicts_and_partition([m],{'bicycle':[other]})
        self.assertEqual(len(review),1)

    def test_same_caption_evidence_resets_and_counter_disable_is_respected(self):
        d = detector(lambda _: [row(.7,'motorcycle'),row(.7,'bicycle')])
        d.config['inference']['compact_group_keys'] = ['contrast_test']
        d.config['inference']['counter_policy'] = 'selective'
        d.prompt_groups['contrast_test'] = {'class_ids':[2], 'box_threshold':.25,
            'distractors':{'bicycle':{'prompt':'bicycle','aliases':['bicycle']}}}
        first = d.predict(Image.new('RGB',(200,200)),query_mode='compact')
        self.assertEqual(len(first['review_candidates']),1)
        disabled = d.predict(Image.new('RGB',(200,200)),query_mode='compact',enable_counter_checks=False)
        self.assertEqual(len(disabled['object_detections']),1)
        self.assertFalse(disabled['counter_evidence'])
        d.processor.responses = lambda _: [row(.7,'motorcycle')]
        next_image = d.predict(Image.new('RGB',(200,200)),query_mode='compact')
        self.assertEqual(len(next_image['object_detections']),1)
        self.assertFalse(next_image['counter_evidence']['bicycle'])

    def test_horizontal_pole_is_retained_as_uncertain_not_deleted(self):
        d = detector(lambda _: [])
        pole = detection('horizontal',8,[0,0,200,30])
        predicted, review, _ = d.analyze_conflicts_and_partition([pole])
        self.assertFalse(predicted)
        self.assertEqual(len(review),1)
        self.assertIn('upright_object_scope_uncertain', review[0].review_reason)

    def test_compact_contrast_predictions_are_debug_invariant(self):
        d = detector(lambda _: [row(.7,'motorcycle'),row(.5,'bicycle'),row(.18,'bicycle')])
        d.config['inference']['compact_group_keys'] = ['contrast_test']
        d.config['inference']['counter_policy'] = 'selective'
        d.prompt_groups['contrast_test'] = {'class_ids':[2], 'box_threshold':.25,
            'distractors':{'bicycle':{'prompt':'bicycle','aliases':['bicycle']}}}
        a = d.predict(Image.new('RGB',(200,200)),query_mode='compact',debug=False)
        b = d.predict(Image.new('RGB',(200,200)),query_mode='compact',debug=True)
        for key in ('object_detections','review_candidates','counter_evidence','counter_plan'):
            self.assertEqual(a[key],b[key],key)
