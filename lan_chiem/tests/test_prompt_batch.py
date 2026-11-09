import unittest
from types import SimpleNamespace
import torch
from lan_chiem.detector import GroundingDINODetector

class PromptBatchTests(unittest.TestCase):
    def test_batch_splits_outputs_in_original_prompt_order(self):
        # Exercise real batching orchestration without loading model weights.
        detector = object.__new__(GroundingDINODetector)
        detector.device = 'cpu'
        calls=[]
        def processor(**kwargs):
            self.assertTrue(kwargs['padding'])
            return {'input_ids':torch.arange(len(kwargs['text'])).reshape(-1,1)}
        detector.processor=processor
        def run(inputs,prompts,source):
            calls.append(prompts)
            ids=inputs['input_ids']
            return SimpleNamespace(logits=ids[:,None,:],pred_boxes=ids[:,None,:].expand(-1,1,4))
        detector._run_model=run
        def crop(**job):
            if '_prepared' in job:
                inputs,outputs=job['_prepared']
                self.assertEqual(inputs['input_ids'].shape[0],1)
                self.assertEqual(outputs.pred_boxes.shape,(1,1,4))
            return [job['prompt_text']]
        detector._infer_crop=crop
        jobs=[{'image_crop':object(),'prompt_text':str(i)} for i in range(5)]
        self.assertEqual(detector._infer_group_batches(jobs,2),list('01234'))
        self.assertEqual(calls,[['0','1'],['2','3']])

