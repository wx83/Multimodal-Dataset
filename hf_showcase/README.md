---
title: AV Data-Curation Agent — Stage Evolution
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# AV Data-Curation Agent — Stage-by-Stage Evolution

Static showcase of how one sample (`9NcjRFu6C4I`) evolves through the
`av_langgraph_pipeline` LangGraph agent:

1. **AV captioning** (Qwen3-Omni)
2. **Sounding-object extraction** → target: *train engine*
3. **Segmentation** (SAM3, text-prompted video masks)
4. **Video object removal** (EffectErase inpainting)
5. **Audio target removal** (SAM-Audio best-of-10, ImageBind-ranked)
6. **Joint AV enhancement** (LTX-2.3 22B denoising enhancer — video **and** audio
   refined together)

Open the Space to see each stage's artifact with inline players, including the
before/after comparison of the LTX-2 enhancement step.
