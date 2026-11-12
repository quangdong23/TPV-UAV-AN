"""Context corrections with controlled hypotheses; no model-learning claim."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from uav_pipeline.detection.postprocess import Detection, analyze_conflicts_and_partition
from uav_pipeline.detection.context import load_context_profile
from uav_pipeline.detection.detector import YOLOEDetector


def hypothesis(identity, cid, box, score=.8):
    labels = {7: 'shop sign', 8: 'utility pole', 9: 'tree trunk', 10: 'packaged products'}
    return Detection(identity, cid, str(cid), str(cid), labels.get(cid, str(cid)), 'yoloe_vocabulary',
                     'full', score, box)


class RecognitionContextTests(unittest.TestCase):
    def setUp(self):
        self.profile, self.metadata = load_context_profile('configs/recognition_context.yaml')
        self.rules = {'target_context': deepcopy(self.profile['runtime_policy']),
                      'spatial_context_review': False}

    def test_pole_below_green_foliage_is_not_rejected_by_background(self):
        image = np.zeros((320, 320, 3), np.uint8)
        image[0:80, :, 1] = 220
        pole = hypothesis('pole', 8, [100, 80, 125, 280])
        predicted, review, _ = analyze_conflicts_and_partition([pole], review_rules=self.rules, surface_image=image)
        self.assertEqual(predicted, [pole])
        self.assertFalse(review)
        self.assertGreater(pole.evidence['canopy_greenness'], 12.)

    def test_diagonal_trunk_and_close_wide_pole_need_more_than_shape(self):
        for cid, box in ((9, [20, 20, 160, 80]), (8, [20, 20, 120, 180])):
            item = hypothesis('object', cid, box, .3)
            predicted, review, _ = analyze_conflicts_and_partition([item], review_rules=self.rules)
            self.assertEqual(predicted, [item])
            self.assertFalse(review)

    def test_strong_same_object_building_column_still_excludes_false_pole(self):
        pole = hypothesis('pole', 8, [20, 20, 120, 180], .4)
        column = {'id': 'column', 'bbox_xyxy': [20, 20, 120, 180], 'score': .9,
                  'origin': 'shared_yoloe_head'}
        predicted, review, _ = analyze_conflicts_and_partition([pole], {'column': [column]}, self.rules)
        self.assertFalse(predicted)
        self.assertIn('upright_structure_competing_column', review[0].review_reason)

    def test_pole_tree_competition_keeps_clear_winner_and_separate_nearby_objects(self):
        pole = hypothesis('pole', 8, [100, 20, 140, 200], .9)
        trunk = hypothesis('trunk', 9, [100, 20, 140, 200], .5)
        predicted, review, _ = analyze_conflicts_and_partition([pole, trunk], review_rules=self.rules)
        self.assertEqual([d.id for d in predicted], ['pole'])
        self.assertEqual([d.id for d in review], ['trunk'])
        pole = hypothesis('pole', 8, [100, 20, 140, 200], .5)
        trunk = hypothesis('trunk', 9, [100, 20, 140, 200], .9)
        predicted, review, _ = analyze_conflicts_and_partition([pole, trunk], review_rules=self.rules)
        self.assertEqual([d.id for d in predicted], ['trunk'])
        pole = hypothesis('pole', 8, [100, 20, 140, 200], .8)
        trunk = hypothesis('trunk', 9, [130, 20, 170, 200], .81)
        predicted, review, _ = analyze_conflicts_and_partition([pole, trunk], review_rules=self.rules)
        self.assertEqual(len(predicted), 2)
        self.assertFalse(review)

    def test_close_pole_tree_scores_remain_unresolved_instead_of_relabelling(self):
        pole = hypothesis('pole', 8, [100, 20, 140, 200], .8)
        trunk = hypothesis('trunk', 9, [100, 20, 140, 200], .81)
        predicted, review, _ = analyze_conflicts_and_partition([pole, trunk], review_rules=self.rules)
        self.assertFalse(predicted)
        self.assertEqual({d.class_id for d in review}, {8, 9})

    def test_bottom_sign_and_explicit_watermark_are_resolution_independent(self):
        for scale in (1., .5):
            box = (np.array([100, 930, 400, 1060])*scale).tolist()
            dimensions = (int(1920*scale), int(1080*scale))
            sign = hypothesis('sign', 7, box)
            predicted, review, _ = analyze_conflicts_and_partition([sign], review_rules=self.rules,
                                                                 image_size=dimensions)
            self.assertEqual(predicted, [sign])
            self.assertFalse(review)
            rules = deepcopy(self.rules)
            rules['target_context']['known_watermark_regions'] = [[0., .8, 1., 1.]]
            sign = hypothesis('overlay', 7, box)
            predicted, review, _ = analyze_conflicts_and_partition([sign], review_rules=rules,
                                                                 image_size=dimensions)
            self.assertFalse(predicted)
            self.assertIn('configured_watermark_region', review[0].review_reason)

    def test_merchandise_can_overlap_its_table_without_becoming_the_table(self):
        goods = hypothesis('goods', 10, [20, 30, 120, 90])
        table = hypothesis('table', 4, [10, 20, 130, 140])
        predicted, review, _ = analyze_conflicts_and_partition([goods, table], review_rules=self.rules)
        self.assertIn(goods, predicted)
        self.assertNotIn(goods, review)
        self.assertEqual(goods.evidence['content_status'], 'content_candidate')
        self.assertEqual(goods.evidence['verification_status'], 'unverified')

    def test_profile_does_not_modify_fused_model_vocabulary(self):
        detector = YOLOEDetector(device='cpu')
        metadata = json.loads((Path(__file__).resolve().parents[1]/'models/yoloe-v8s-single.json').read_text())
        self.assertEqual([list(row) for row in detector.vocabulary], metadata['vocabulary'])
        self.assertEqual(detector.vocabulary_sha256, metadata['source']['vocabulary_sha256'])
        self.assertFalse(detector.config['recognition_context']['text_descriptions_change_fused_embeddings'])
        for entry in self.profile['classes'].values():
            self.assertGreaterEqual(len(entry['positive_prompts']), 10)
            self.assertTrue(entry['common_confusions'])

    def test_invalid_context_policy_is_rejected(self):
        for value in (float('nan'), 0, 2, True):
            profile = deepcopy(self.profile)
            profile['runtime_policy']['competing_score_margin'] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'context.yaml'
                path.write_text(yaml.safe_dump(profile, allow_unicode=True))
                with self.assertRaises(ValueError):
                    load_context_profile(path)


class FocusCounterPrecisionTests(unittest.TestCase):
    setUp = RecognitionContextTests.setUp

    def test_sign_and_goods_clear_same_object_counter_requires_review(self):
        for cid, group in ((7, 'traffic_sign'), (7, 'canopy'),
                           (10, 'potted_plant'), (10, 'construction_materials')):
            target = hypothesis('target', cid, [20, 20, 120, 180], .4)
            counter = {'id': 'counter', 'bbox_xyxy': [20, 20, 120, 180],
                       'score': .8, 'origin': 'shared_yoloe_head', 'counter_group': group}
            predicted, review, _ = analyze_conflicts_and_partition(
                [target], {group: [counter]}, self.rules)
            self.assertFalse(predicted)
            self.assertEqual(review, [target])

    def test_near_tie_keeps_target_when_counter_not_clearly_stronger(self):
        for cid, group in ((7, 'traffic_sign'), (10, 'potted_plant')):
            target = hypothesis('target', cid, [20, 20, 120, 180], .8)
            counter = {'id': 'counter', 'bbox_xyxy': [20, 20, 120, 180],
                       'score': .82, 'origin': 'shared_yoloe_head'}
            predicted, review, _ = analyze_conflicts_and_partition(
                [target], {group: [counter]}, self.rules)
            self.assertEqual(predicted, [target])
            self.assertFalse(review)

    def test_small_plant_inside_goods_and_canopy_behind_sign_are_context(self):
        for cid, group in ((7, 'canopy'), (10, 'potted_plant')):
            target = hypothesis('target', cid, [20, 20, 120, 180], .4)
            counter = {'id': 'counter', 'bbox_xyxy': [30, 30, 50, 70],
                       'score': .9, 'origin': 'shared_yoloe_head'}
            predicted, review, _ = analyze_conflicts_and_partition(
                [target], {group: [counter]}, self.rules)
            self.assertEqual(predicted, [target])
            self.assertFalse(review)

    def test_clear_target_winner_is_kept(self):
        target = hypothesis('target', 10, [20, 20, 120, 180], .8)
        counter = {'id': 'counter', 'bbox_xyxy': [20, 20, 120, 180],
                   'score': .5, 'origin': 'shared_yoloe_head'}
        predicted, review, _ = analyze_conflicts_and_partition(
            [target], {'potted_plant': [counter]}, self.rules)
        self.assertEqual(predicted, [target])
        self.assertFalse(review)


class RelaxedProductionConfigTests(unittest.TestCase):
    def test_sign_counters_are_disabled_in_production(self):
        detector = YOLOEDetector(device='cpu')
        rules = detector.config['review_rules']
        self.assertFalse(rules['advertising_traffic_overlap'])
        self.assertFalse(rules['sign_canopy_overlap'])
        sign = hypothesis('sign', 7, [20, 20, 120, 180], .4)
        counter = {'id': 'counter', 'bbox_xyxy': [20, 20, 120, 180],
                   'score': .9, 'origin': 'shared_yoloe_head'}
        predicted, review, _ = analyze_conflicts_and_partition(
            [sign], {'traffic_sign': [counter], 'canopy': [counter]}, rules)
        self.assertEqual(predicted, [sign])
        self.assertFalse(review)


class DirectFixRegressionTests(unittest.TestCase):
    def test_weak_pole_cannot_remove_comparable_trunk(self):
        profile, _ = load_context_profile('configs/recognition_context.yaml')
        rules = {'target_context': profile['runtime_policy'], 'weak_pole_tree_policy': True}
        for reverse in (False, True):
            pole = hypothesis('pole', 8, [100,20,140,200], .30)
            trunk = hypothesis('trunk', 9, [100,20,140,200], .22)
            items = [pole,trunk] if not reverse else [trunk,pole]
            predicted, review, _ = analyze_conflicts_and_partition(items, review_rules=rules)
            self.assertEqual(predicted, [trunk])
            self.assertEqual(review, [pole])
            self.assertEqual(trunk.evidence['confirmation_hits'], 4)

    def test_strong_pole_and_nearby_trunk_are_preserved(self):
        profile, _ = load_context_profile('configs/recognition_context.yaml')
        rules = {'target_context': profile['runtime_policy'], 'weak_pole_tree_policy': True}
        pole = hypothesis('pole', 8, [100,20,140,200], .8)
        trunk = hypothesis('trunk', 9, [145,20,185,200], .3)
        predicted, review, _ = analyze_conflicts_and_partition([pole,trunk], review_rules=rules)
        self.assertEqual(len(predicted),2)
        self.assertFalse(review)

    def test_weak_trunk_needs_extent_strong_diagonal_does_not(self):
        rules = {'weak_trunk_candidates': True, 'upright_object_geometry': False}
        weak = hypothesis('weak',9,[10,10,30,200],.08)
        tiny = hypothesis('tiny',9,[100,100,115,130],.08)
        diagonal = hypothesis('diagonal',9,[100,100,280,140],.7)
        predicted, review, _ = analyze_conflicts_and_partition([weak,tiny,diagonal],review_rules=rules,image_size=(640,480))
        self.assertIn(weak,predicted)
        self.assertIn(diagonal,predicted)
        self.assertIn(tiny,review)
        self.assertEqual(weak.evidence['confirmation_hits'],4)

    def test_packaged_goods_guard_does_not_change_fruit_threshold(self):
        rules = {'phrase_review_thresholds': {'stacked packaged goods': .45}}
        packaged = hypothesis('packaged',10,[10,10,50,50],.3)
        packaged.raw_label='stacked packaged goods'
        fruit = hypothesis('fruit',10,[60,10,100,50],.3)
        fruit.raw_label='basket of fruit'
        predicted,review,_=analyze_conflicts_and_partition([packaged,fruit],review_rules=rules)
        self.assertEqual(predicted,[fruit])
        self.assertEqual(review,[packaged])

    def test_explicit_road_probability_is_required_and_high_confidence_is_kept(self):
        from uav_pipeline.detection.postprocess import partition_road_surface_hypotheses
        rules={'road_surface_consistency':True}
        rows=[hypothesis('marking',8,[10,10,30,90],.28).to_dict(),
              hypothesis('strong',8,[40,10,60,90],.8).to_dict()]
        for road in (None,np.zeros((100,100),np.uint8),np.full((100,100),np.nan)):
            kept,review=partition_road_surface_hypotheses(rows,road,rules)
            self.assertEqual(len(kept),2)
            self.assertFalse(review)
        kept,review=partition_road_surface_hypotheses(rows,np.ones((100,100)),rules)
        self.assertEqual([d['id'] for d in kept],['strong'])
        self.assertEqual([d['id'] for d in review],['marking'])
        self.assertEqual(rows[0]['status'],'predicted')

    def test_weak_tree_needs_four_detector_hits_flow_does_not_confirm(self):
        from uav_pipeline.tracking.objects import ObjectTracker
        from tests.test_tracking_trunks import motion
        tracker=ObjectTracker(min_hits=2)
        row=hypothesis('weak',9,[10,10,30,200],.08).to_dict()
        row['evidence']={'confirmation_hits':4}
        for i in range(3):
            self.assertFalse(tracker.update(i*.1,[row],motion()))
            self.assertFalse(tracker.update(i*.1+.01,None,motion()))
        self.assertEqual(len(tracker.update(.3,[row],motion())),1)

    def test_strong_confirmed_track_does_not_become_tentative_on_weak_hit(self):
        from uav_pipeline.tracking.objects import ObjectTracker
        from tests.test_tracking_trunks import motion
        tracker=ObjectTracker(min_hits=2)
        row=hypothesis('strong',9,[10,10,30,200],.8).to_dict()
        tracker.update(0,[row],motion())
        self.assertEqual(len(tracker.update(.1,[row],motion())),1)
        row['score']=.08;row['evidence']={'confirmation_hits':4}
        self.assertEqual(len(tracker.update(.2,[row],motion())),1)


class FinalGuardTests(unittest.TestCase):
    def test_partial_road_box_and_real_sign_other_surface(self):
        from uav_pipeline.detection.postprocess import partition_road_surface_hypotheses
        road=np.ones((100,100));road[:10,:]=0
        rows=[hypothesis('shadow',7,[0,0,100,100],.32).to_dict()]
        kept,review=partition_road_surface_hypotheses(rows,road,{'road_surface_consistency':True})
        self.assertFalse(kept)
        self.assertEqual(review[0]['review_reason'],'weak_object_on_drivable_surface')
        kept,review=partition_road_surface_hypotheses(rows,np.zeros((100,100)),{'road_surface_consistency':True})
        self.assertEqual([d['id'] for d in kept], [d['id'] for d in rows])
        self.assertEqual(kept[0]['status'], 'predicted')
        self.assertEqual(kept[0]['evidence']['road_surface_fraction'], 0.)
        self.assertEqual(rows[0]['evidence'], {})
        self.assertFalse(review)

    def test_clear_strong_trunk_winner_is_not_marked_uncertain(self):
        profile,_=load_context_profile('configs/recognition_context.yaml')
        rules={'target_context':profile['runtime_policy'],'weak_pole_tree_policy':True}
        pole=hypothesis('pole',8,[100,20,140,200],.3)
        tree=hypothesis('tree',9,[100,20,140,200],.8)
        predicted,review,_=analyze_conflicts_and_partition([pole,tree],review_rules=rules)
        self.assertEqual(predicted,[tree])
        self.assertNotIn('pole_tree_uncertain',tree.evidence)


class PavementBoundaryTests(unittest.TestCase):
    def test_weak_curb_line_requires_surface_evidence_but_pole_is_kept(self):
        from uav_pipeline.detection.postprocess import partition_road_surface_hypotheses
        rows=[hypothesis('curb',8,[10,10,15,95],.3).to_dict()]
        road=np.full((100,100),.6);sidewalk=np.full((100,100),.38)
        kept,review=partition_road_surface_hypotheses(rows,road,{'road_surface_consistency':True},sidewalk)
        self.assertFalse(kept)
        self.assertEqual(review[0]['review_reason'],'weak_pole_on_surface_boundary')
        # Same geometry, but a raised-object region has substantial non-ground evidence.
        kept,review=partition_road_surface_hypotheses(rows,road*.2,{'road_surface_consistency':True},sidewalk*.2)
        self.assertEqual(len(kept),1)
        self.assertFalse(review)
        # Boundary suppression is never applied to signs or trunks.
        rows[0]['class_id']=9
        kept,review=partition_road_surface_hypotheses(rows,road,{'road_surface_consistency':True},sidewalk)
        self.assertEqual(len(kept),1)
        self.assertFalse(review)


class WeakTrunkBodyTests(unittest.TestCase):
    def test_dark_body_recovery_does_not_promote_plain_bright_pole(self):
        rules={'weak_trunk_candidates':True,'upright_object_geometry':False}
        for body,expected in ((40,True),(180,False)):
            image=np.full((480,640,3),180,np.uint8)
            image[20:300,105:135]=body
            item=hypothesis('candidate',9,[100,20,140,300],.08)
            predicted,review,_=analyze_conflicts_and_partition([item],review_rules=rules,surface_image=image)
            self.assertEqual(bool(predicted),expected)
            self.assertEqual(bool(review),not expected)
        # This recovery guard never rejects an already strong pale/painted trunk.
        item=hypothesis('painted',9,[100,20,140,300],.7)
        predicted,review,_=analyze_conflicts_and_partition([item],review_rules=rules,surface_image=image)
        self.assertEqual(predicted,[item])
        self.assertFalse(review)
