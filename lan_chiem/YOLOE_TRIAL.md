# YOLOE-v8-S trial

The image CLI now defaults to `yoloe-v8s-seg.pt`. Install dependencies in the
environment where inference runs:

```bash
python -m pip install -r lan_chiem/requirements-yoloe.txt
python -m lan_chiem.predict --backend yoloe --input lan_chiem/sample_data/1.png --output lan_chiem/outputs/yoloe_trial --device auto
```

The first invocation downloads the checkpoint and text encoder assets. Prompt
embeddings are computed once per detector instance. Image timing excludes model
loading and embedding setup; process wall time includes them.

This experimental backend performs one full-image YOLOE pass at 640 pixels,
maps vocabulary entries to catalogue IDs 1–10, applies per-class config thresholds,
intra-class NMS and the shared conflict rules, and writes annotated JPG and JSON.
It does not reproduce DINO tiling, cart merging, leg/tree filters or custom prompt
inspection. DINO-specific overrides are rejected. Accuracy is unverified on UAV data.

Return to DINO Tiny with `--backend grounding-dino`.

Video CLI also accepts `--backend yoloe --weights yoloe-v8s-seg.pt`.
The existing Colab notebook has not been switched to YOLOE.
