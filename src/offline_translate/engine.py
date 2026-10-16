"""Runs one direction of a Google Translate on-device NMT model (TFLite encoder + LSTM decoder)."""
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import sentencepiece as spm
from ai_edge_litert.interpreter import Interpreter

BOS, EOS = 1, 2
STATE_KEYS = [f"beam_state:rnn_state_{i}_{x}:00" for i in (0, 1) for x in ("c", "m")]
# The models are trained on single sentences; split on sentence-final punctuation + whitespace.
SENT_RE = re.compile(r"(?<=[.!?…。！？])\s+")


def split_sentences(text):
    """Lines of sentences: line breaks are kept, each line is split into sentences."""
    return [[s for s in SENT_RE.split(line.strip()) if s] for line in text.split("\n")]


FINAL_PUNCT = ".!?…。！？:;\"'”’)»"


def add_period_limit(processor_spec):
    """max_sent_len of the pack's PreprocAddPeriod step, or None if the pack doesn't use it."""
    m = re.search(rb"PreprocAddPeriod/[^:]*:max_sent_len=(\d+)", processor_spec)
    return int(m.group(1)) if m else None


def with_period(sentence, limit):
    """Short sentences without final punctuation translate badly ("car" -> "tram"); the app adds
    a period first. Returns (text to translate, whether a period was added)."""
    if limit and len(sentence.split()) <= limit and sentence[-1:] not in FINAL_PUNCT:
        return sentence + ".", True
    return sentence, False


def strip_period(translation):
    return translation[:-1] if translation[-1:] in ".。" else translation


class NmtModel:
    def __init__(self, path):
        path = Path(path)
        self.src_sp = spm.SentencePieceProcessor(model_file=str(path / "src_spm_model.model"))
        self.tgt_sp = spm.SentencePieceProcessor(model_file=str(path / "tgt_spm_model.model"))
        runner = lambda name: Interpreter(model_path=str(path / f"{name}.tflite")).get_signature_runner()
        self.encoder = runner("encoder_0")
        self.dec_init = runner("decoder_init_0")
        self.dec_step = runner("decoder_step_0")
        spec = path / "processor_spec.pb"
        self.period_limit = add_period_limit(spec.read_bytes()) if spec.exists() else None

    def translate(self, text, beam=4):
        return "\n".join(" ".join(self.translate_sentence(s, beam=beam) for s in line)
                         for line in split_sentences(text))

    def translate_sentence(self, text, beam=4):
        text, added = with_period(text, self.period_limit)
        out = self._decode(text, beam=beam)
        return strip_period(out) if added else out

    def _decode(self, text, beam=4, max_len=72, alpha=1.0):
        # Source vocabularies are lowercase-only; the model restores casing on output.
        ids = self.src_sp.encode(text.lower()) + [EOS]
        src = np.array([ids], np.int32)
        pad = np.zeros_like(src, np.float32)
        encs = self.encoder(**{"feeds:src_ids:00": src, "feeds:src_paddings:00": pad})["fetches:src_encs:00"]
        init = self.dec_init(**{
            "feeds:num_hyps:00": np.array([1], np.int32),
            "feeds:src_enc_paddings:00": pad,
            "feeds:src_encs:00": encs,
        })
        vecs = init["atten_init:source_vecs:00"]
        state = {k: np.zeros((1, 512), np.float32) for k in STATE_KEYS}
        state["beam_state:atten_context:00"] = init["beam_state:atten_context:00"]

        # Each live hypothesis: (tokens, cumulative log prob). Batch rows of `state` align with `hyps`.
        hyps = [([], 0.0)]
        done = []
        max_len = min(max_len, 2 * len(ids) + 10)
        for _ in range(max_len):
            n = len(hyps)
            last = np.array([[t[-1] if t else BOS] for t, _ in hyps], np.int32)
            out = self.dec_step(**{
                "atten_init:source_contexts:00": np.repeat(vecs, n, 0),
                "atten_init:source_vecs:00": np.repeat(vecs, n, 0),
                "atten_init:source_padding:00": np.repeat(pad, n, 0),
                "feeds:step_ids:00": last,
                **state,
            })
            logp = out["fetches:log_probs:00"]
            scores = np.array([s for _, s in hyps])[:, None] + logp
            flat = np.argsort(scores, axis=None)[::-1][: 2 * beam]
            new_hyps, rows = [], []
            for idx in flat:
                row, tok = divmod(int(idx), logp.shape[1])
                toks, sc = hyps[row][0] + [tok], float(scores[row, tok])
                if tok == EOS:
                    done.append((toks[:-1], sc / (((5 + len(toks)) / 6) ** alpha)))
                elif len(new_hyps) < beam:
                    new_hyps.append((toks, sc))
                    rows.append(row)
            if not new_hyps:
                break
            # Stop when the best finished hypothesis beats any live one's optimistic score.
            if len(done) >= beam:
                best_done = max(s for _, s in done)
                best_live = max(s for _, s in new_hyps) / (((5 + max_len) / 6) ** alpha)
                if best_done > best_live:
                    break
            hyps = new_hyps
            state = {k: out[k][rows] for k in state}
        if not done:
            done = hyps
        return self.tgt_sp.decode(max(done, key=lambda h: h[1])[0])
