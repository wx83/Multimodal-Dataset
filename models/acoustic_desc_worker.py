"""Acoustic description of a clip: what is actually sounding, as a short label.

Runs in an env with torch + laion_clap (on point.dd.works: ~/miniconda3/envs/sam3).
Invoked as a subprocess by acoustic_desc_model.py — never imported by the pipeline.

Why (strategy-lab S49/S50/S55/S56): the extraction node emits one VISUAL noun
(object_name). For 90% of delivered person-class samples that noun is "man"/
"woman"/"person" while the clip contains no speech — the sound present is the
person's actions (footsteps, door closing, dishes). Prompting the separator and
scoring removal with "man" gives no signal; the zero-shot acoustic label does
(AUC 0.496 -> 0.634). This worker produces that label from the ORIGINAL audio,
before separation, so it does not need a caption or an LLM.

Zero-shot CLAP over the same 130-label table as clap_select_worker.py.
Writes a JSON result to --out and a marker line to stdout.
"""

import argparse
import json
import os
import sys

RESULT_MARKER = "ACOUSTIC_DESC_RESULT "
DEFAULT_CKPT = os.environ.get("CLAP_CKPT", "/scratch/weihan/630k-audioset-best.pt")
DEFAULT_TEMPLATE = "the sound of {}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="wav/mp4 with the ORIGINAL (pre-separation) audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--target", default=None,
                    help="object_name; for a person-class target speech labels are excluded (owner decision, A02)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from clap_select_worker import ACOUSTIC_LABELS, SPEECH_LABELS, is_person, pick_acoustic_label
    import numpy as np
    import soundfile as sf
    import torch

    _orig_load = torch.load

    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _load
    import laion_clap

    x, sr = sf.read(args.audio, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 48000:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 48000).numpy()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=device)
    model.load_ckpt(args.ckpt)
    model.eval()

    def l2n(m):
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)
    with torch.no_grad():
        temb = l2n(np.asarray(model.get_text_embedding(
            [args.template.format(l) for l in ACOUSTIC_LABELS], use_tensor=False)))
        aemb = l2n(np.asarray(model.get_audio_embedding_from_data(
            x=x[None, :].astype(np.float32), use_tensor=False)))[0]
    sims = aemb @ temb.T
    person = is_person(args.target)
    label, score = pick_acoustic_label(sims, exclude=SPEECH_LABELS if person else ())
    order = np.argsort(-sims)[:args.top_k]
    # speech presence is reported on its own: it is not the target, but a
    # speech+action sample is the richest kind (owner, A02) and worth marking
    speech_score = max(float(sims[i]) for i in range(len(ACOUSTIC_LABELS)) if ACOUSTIC_LABELS[i] in SPEECH_LABELS)
    res = {
        "audio": args.audio,
        "target": args.target,
        "acoustic_desc": label,
        "acoustic_desc_score": score,
        "speech_excluded": person,
        "speech_score": speech_score,
        "top": [{"label": ACOUSTIC_LABELS[i], "score": float(sims[i])} for i in order],
        "source": "clap-zeroshot",
        "ckpt": args.ckpt,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
