"""Do not silently promote a fast model whose reference comparison failed."""


def require_candidate_audit(backend, report, source_sha256, weights, imgsz, work_max_side):
    if backend == 'grounding-dino':
        return
    if backend != 'world':
        raise ValueError(f'Unknown backend: {backend}')
    if not report or report.get('agreement_screen_passed') is not True:
        raise ValueError('Candidate audit missing/failed. Do not run the fast candidate as an accepted replacement.')
    expected = {'source_sha256': source_sha256, 'weights': str(weights),
                'imgsz': imgsz, 'work_max_side': work_max_side, 'backend': 'yolo_world'}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Audit does not match this video/model/resolution. Re-run the audit.')
