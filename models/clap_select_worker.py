"""CLAP best-candidate selection for SAM-Audio best_of outputs.

Alternative to ib_select_worker.py, enabled with BESTOF_SELECTOR=clap (default
stays ib — see audio_removal_model.py). Runs inside an env that has torch +
laion_clap (on point.dd.works: ~/miniconda3/envs/sam3). Invoked as a
subprocess — never imported by the pipeline, except for rank_candidates(),
which is pure Python so the ranking rule can be unit-tested without torch.

Why this exists (avgraph-strategy-lab S49/S52/S53): the ImageBind key
(max ib_ta, tiebreak min ib_ta_res) measures whether the separated TARGET
sounds like the prompt, not whether the prompt's sound left the RESIDUAL.
Within a sample its rank correlation with removal quality is +0.02 (307
samples x 10 candidates); no re-keying of the two ImageBind numbers recovers
the gain. Scoring each candidate with CLAP text<->audio on mix vs residual
does: removal = s_mix - s_res, +0.047 median (upper bound), and on an
independent VAD measure 7 improved / 2 worse of 20 speech samples.

Scores every candidate with:
  s_mix, s_target, s_res : CLAP cosine(text, mix / target / residual), mix = target + residual
  clap_removal           : s_mix - s_res   (higher = the prompt's sound left the residual)
  residual_energy        : rms(residual) / rms(mix)  (guards against "delete everything")
Selects max clap_removal among candidates with residual_energy >= --min-residual-energy
(falls back to all candidates if none qualify), tiebreak higher s_target.

Writes a JSON result to --out and a marker line to stdout.
"""

import argparse
import json
import os
import shutil
import sys

RESULT_MARKER = "CLAP_SELECT_RESULT "
DEFAULT_CKPT = os.environ.get("CLAP_CKPT", "/scratch/weihan/630k-audioset-best.pt")
DEFAULT_TEMPLATE = "the sound of {}"
DEFAULT_MIN_RESIDUAL_ENERGY = 0.3


def rank_candidates(rows, min_residual_energy=DEFAULT_MIN_RESIDUAL_ENERGY):
    """Pure ranking rule. Each row needs clap_removal, residual_energy, s_target.

    Returns rows sorted best-first. Candidates whose residual kept less than
    `min_residual_energy` of the mix energy are moved to the back regardless of
    score: a near-silent residual maximises "removal" by deleting everything,
    which is the reward-hacking shape frozen_judge warned about.
    """
    def key(r):
        ok = r["residual_energy"] >= min_residual_energy
        return (0 if ok else 1, -r["clap_removal"], -r.get("s_target", 0.0))
    return sorted(rows, key=key)


def _score(rows, text, ckpt, template, device):
    import numpy as np
    import soundfile as sf
    import torch

    # torch>=2.6 defaults to weights_only=True; the LAION checkpoint pickles numpy scalars.
    _orig_load = torch.load

    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_load(*a, **kw)
    torch.load = _load
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=device)
    model.load_ckpt(ckpt)
    model.eval()

    def l2n(m):
        return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)

    def wav(p):
        x, _ = sf.read(p, dtype="float32")
        return x.mean(axis=1) if x.ndim > 1 else x

    with torch.no_grad():
        temb = l2n(np.asarray(model.get_text_embedding([template.format(text)], use_tensor=False)))[0]
    for r in rows:
        tg, rs = wav(r["target_wav"]), wav(r["residual_wav"])
        n = min(len(tg), len(rs)); tg, rs = tg[:n], rs[:n]
        mix = np.clip(tg + rs, -1.0, 1.0)
        with torch.no_grad():
            ae = l2n(np.asarray(model.get_audio_embedding_from_data(
                x=np.stack([mix, tg, rs]).astype(np.float32), use_tensor=False)))
        s = ae @ temb
        rms = lambda v: float(np.sqrt((v ** 2).mean()))
        r["s_mix"], r["s_target"], r["s_res"] = float(s[0]), float(s[1]), float(s[2])
        r["clap_removal"] = r["s_mix"] - r["s_res"]
        r["residual_energy"] = rms(rs) / max(rms(mix), 1e-9)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="candidates.json from sam_audio_worker.py --mode best_of")
    ap.add_argument("--text", default=None, help="Prompt text; defaults to the candidates.json prompt.")
    ap.add_argument("--out", default=None, help="Where to write selection.json.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--min-residual-energy", type=float, default=DEFAULT_MIN_RESIDUAL_ENERGY)
    args = ap.parse_args()

    with open(args.candidates, encoding="utf-8") as f:
        cand = json.load(f)
    rows = cand["rows"]
    text = args.text or cand["prompt"]
    out_dir = os.path.dirname(os.path.abspath(args.candidates))
    sample_id = cand.get("sample_id", "sample")

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = _score(rows, text, args.ckpt, args.template, device)
    ranked = rank_candidates(rows, args.min_residual_energy)
    best = ranked[0]
    best_target = os.path.join(out_dir, f"{sample_id}_best_target.wav")
    best_residual = os.path.join(out_dir, f"{sample_id}_best_residual.wav")
    shutil.copyfile(best["target_wav"], best_target)
    shutil.copyfile(best["residual_wav"], best_residual)

    res = {
        "sample_id": sample_id,
        "text": text,
        "video_path": cand.get("video_path"),
        "selector": "clap",
        "selector_config": {"ckpt": args.ckpt, "template": args.template,
                            "min_residual_energy": args.min_residual_energy},
        "best": {**best, "best_target": best_target, "best_residual": best_residual},
        "candidates": ranked,
    }
    out_path = args.out or os.path.join(out_dir, "selection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    sys.stdout.write(RESULT_MARKER + json.dumps(res, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
