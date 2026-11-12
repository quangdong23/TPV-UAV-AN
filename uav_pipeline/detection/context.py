"""Explicit runtime policy and offline guidance; never rewrites fused vocabulary."""
import hashlib
from pathlib import Path

import yaml


from uav_pipeline.paths import PROJECT_ROOT as ROOT


def load_context_profile(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT/path
    data = path.read_bytes()
    profile = yaml.safe_load(data)
    if (not isinstance(profile, dict) or profile.get('schema_version') != 1
            or profile.get('semantic_mode') != 'fixed_vocabulary_runtime_rules_and_offline_annotation_guidance'
            or not isinstance(profile.get('classes'), dict)
            or set(profile.get('classes', {})) != {7, 8, 9, 10}):
        raise ValueError('Recognition context must declare schema 1 and catalog classes 7,8,9,10')
    for entry in profile['classes'].values():
        for key in ('variants', 'supporting_cues', 'common_confusions', 'positive_prompts',
                    'hard_negative_prompts', 'annotation_rules'):
            values = entry.get(key)
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f'Recognition context requires nonempty {key}')
    policy = profile.get('runtime_policy', {})
    if not isinstance(policy, dict):
        raise ValueError('Context runtime_policy must be a mapping')
    for key in ('enabled', 'geometry_requires_competing_evidence',
                'green_background_is_not_negative_evidence', 'pole_tree_same_object_only'):
        if type(policy.get(key)) is not bool:
            raise ValueError(f'Context policy {key} must be boolean')
    for key in ('competing_score_margin', 'competing_iou', 'competing_mutual_coverage',
                'watermark_min_box_coverage'):
        value = policy.get(key)
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError(f'Context policy {key} must be in (0,1]')
    regions = policy.get('known_watermark_regions')
    if not isinstance(regions, list):
        raise ValueError('Watermarks must be explicit normalized rectangles')
    for region in regions:
        if (not isinstance(region, list) or len(region) != 4
                or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in region)
                or region[2] <= region[0] or region[3] <= region[1]):
            raise ValueError('Invalid normalized watermark rectangle')
    return profile, {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
                     'schema_version': 1, 'semantic_mode': profile['semantic_mode'],
                     'text_descriptions_change_fused_embeddings': False}
