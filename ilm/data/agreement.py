"""Subject-verb number agreement — a classic syntactic-circuit behavior (Linzen 2016;
Lakretz 2019; Finlayson 2021). The model must track the grammatical NUMBER of the head
subject across an intervening ATTRACTOR of the opposite number, then predict the
agreeing verb.

2026-07-03 redesign: the original version had exactly ONE syntactic template (a single
prepositional-phrase attractor) — real lexical variety, zero STRUCTURAL variety. The
number-agreement literature studies this behavior specifically by varying the SHAPE of
the intervening material, not just the words in it, so we now generate a MIX of three
frames drawn from that literature:

  "pp1" (Linzen 2016, the original template): one PP attractor.
      clean:  "The {N1p} {prep} the {N2s}"              -> " are"/" were"
      corrupt:"The {N1s} {prep} the {N2p}"               -> " is"/" was"

  "pp2" (Linzen 2016's attractor-COUNT axis): two chained PP attractors, testing
  whether a second opposite-number distractor compounds the difficulty.
      clean:  "The {N1p} {prep1} the {N2s} {prep2} the {N3s}"   -> " are"/" were"
      corrupt:"The {N1s} {prep1} the {N2p} {prep2} the {N3p}"   -> " is"/" was"

  "rc" (Marvin & Linzen 2018's subject-relative-clause construction; also the shape
  Lakretz et al. 2019 use for nested-dependency number tracking): the attractor sits
  inside a relative clause modifying the head, so the model must track the head's
  number PAST an embedded clause with its own (head-agreeing) verb, not just a flat PP.
      clean:  "The {N1p} that {Vrel_p} the {N2s}"        -> " are"/" were"
      corrupt:"The {N1s} that {Vrel_s} the {N2p}"        -> " is"/" was"

All three keep the SAME clean/corrupted discipline as before: every swapped slot (every
noun, every verb) is filtered to a single GPT-2/Qwen/Llama token before it's ever used,
so a length-match between clean_prompt and corrupted_prompt is (as before) equivalent to
"only the intended slots differ" — the same guarantee, just now covering more slots per
frame. Single-token final verbs (" is"/" are", " was"/" were"), so the IOI/docstring
single-token accuracy metric applies unchanged regardless of frame.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from torch.utils.data import Dataset

from .dataset import bos_prefix

# (singular, plural) noun pairs; filtered to single-token forms per tokenizer at build time.
NOUN_PAIRS = [
    ("key", "keys"), ("dog", "dogs"), ("cat", "cats"), ("car", "cars"), ("book", "books"),
    ("door", "doors"), ("road", "roads"), ("tree", "trees"), ("star", "stars"), ("bird", "birds"),
    ("box", "boxes"), ("girl", "girls"), ("boy", "boys"), ("king", "kings"), ("ship", "ships"),
    ("lake", "lakes"), ("hill", "hills"), ("song", "songs"), ("game", "games"),
    ("chair", "chairs"), ("table", "tables"), ("phone", "phones"), ("coin", "coins"),
    ("flag", "flags"), ("guard", "guards"), ("farmer", "farmers"), ("teacher", "teachers"),
    ("painter", "painters"), ("singer", "singers"), ("writer", "writers"), ("doctor", "doctors"),
    ("lawyer", "lawyers"), ("driver", "drivers"), ("player", "players"), ("winner", "winners"),
    ("visitor", "visitors"), ("neighbor", "neighbors"), ("author", "authors"),
    ("artist", "artists"), ("actor", "actors"),
]
PREPS = ["near", "by", "behind", "beside", "under", "above"]
# (singular_verb, plural_verb) for the FINAL predicted answer — copula, auxiliary ("has"/
# "have", "does"/"do" — the two verb families Linzen 2016 tests, beyond copula), and a set
# of ordinary present-tense verbs so the answer isn't always "is/are/was/were" (both single
# token with a leading space, filtered per tokenizer at build time).
VERB_PAIRS = [
    (" is", " are"), (" was", " were"), (" has", " have"), (" does", " do"),
    (" seems", " seem"), (" remains", " remain"), (" appears", " appear"),
    (" waits", " wait"), (" arrives", " arrive"), (" stays", " stay"),
    (" belongs", " belong"), (" matters", " matter"), (" looks", " look"),
    (" sounds", " sound"), (" feels", " feel"), (" works", " work"),
    (" happens", " happen"), (" exists", " exist"), (" continues", " continue"),
    (" begins", " begin"), (" ends", " end"), (" starts", " start"),
    (" stops", " stop"), (" grows", " grow"), (" shows", " show"),
    (" changes", " change"), (" moves", " move"), (" turns", " turn"),
    (" falls", " fall"), (" rises", " rise"), (" breaks", " break"),
    (" opens", " open"), (" closes", " close"), (" runs", " run"),
    (" sits", " sit"), (" stands", " stand"), (" shines", " shine"),
    (" burns", " burn"), (" shakes", " shake"), (" spins", " spin"),
    (" floats", " float"), (" sinks", " sink"), (" returns", " return"),
    (" approaches", " approach"), (" passes", " pass"), (" leads", " lead"),
    (" follows", " follow"), (" helps", " help"), (" fades", " fade"),
    (" glows", " glow"), (" drifts", " drift"),
]
# Transitive present-tense verb pairs for the "rc" frame's embedded relative-clause verb
# (agrees with the HEAD noun, not the attractor) — (singular, plural), single token.
RC_VERB_PAIRS = [
    (" watches", " watch"), (" chases", " chase"), (" follows", " follow"),
    (" helps", " help"), (" finds", " find"), (" hears", " hear"), (" holds", " hold"),
    (" likes", " like"), (" knows", " know"), (" sees", " see"), (" wants", " want"),
    (" needs", " need"), (" visits", " visit"), (" calls", " call"), (" joins", " join"),
]
FRAMES = ("pp1", "pp2", "rc")


@dataclass
class AgreementSample:
    clean_prompt: str       # plural head -> answer " are"/" were"
    corrupted_prompt: str   # singular head (all slots flipped), token-aligned
    answer: str             # the agreeing verb for the CLEAN prompt (single token)
    frame: str = "pp1"      # which construction generated this sample (provenance only)
    plural: bool = True     # is the CLEAN subject plural? selects the valid-answer SET
                            # (see valid_verb_ids) -- NOT which exact verb was drawn.
    contrast: str = ""      # the SAME verb in the WRONG number (the minimal-pair foil for
                            # Linzen 2016 forced-choice scoring: correct iff the model puts
                            # more probability on ``answer`` than on this opposite-number form).


def valid_verb_ids(tokenizer, plural: bool, verbs=None) -> list[int]:
    """Token ids of every verb form matching the given NUMBER (plural or singular),
    across the whole verb pool -- the 'argmax-valid' set for accuracy scoring. With a
    49-pair verb pool, no model (trained or not) can guess WHICH exact synonym a given
    example arbitrarily drew (nothing in the prompt disambiguates "seems" from "remains"
    from "are") -- confirmed empirically: base GPT-2's top-1 matched the literal intended
    answer 0/8 times on a sample. Accuracy must therefore check "did the model produce
    ANY grammatically correct verb of the right number", exactly like greater-than's
    argmax-valid metric checks membership in a set of valid years, not one exact token."""
    pool = verbs if verbs is not None else VERB_PAIRS
    idx = 1 if plural else 0
    ids = set()
    for pair in pool:
        form = pair[idx]
        if _single_tok(tokenizer, form):
            ids.add(tokenizer(form, add_special_tokens=False).input_ids[0])
    return sorted(ids)


def _single_tok(tokenizer, text: str) -> bool:
    return len(tokenizer(text, add_special_tokens=False).input_ids) == 1


def _filter_nouns(tokenizer):
    out = []
    for s, p in NOUN_PAIRS:
        if _single_tok(tokenizer, " " + s) and _single_tok(tokenizer, " " + p):
            out.append((s, p))
    return out


def _filter_verbs(tokenizer, pairs):
    return [(s, p) for s, p in pairs if _single_tok(tokenizer, s) and _single_tok(tokenizer, p)]


def _aligned(tokenizer, clean_prompt: str, corr_prompt: str) -> bool:
    return (len(tokenizer(clean_prompt, add_special_tokens=False).input_ids)
            == len(tokenizer(corr_prompt, add_special_tokens=False).input_ids))


def build_samples(tokenizer, n: int, seed: int = 0,
                  exclude: set[str] | None = None,
                  frames: tuple[str, ...] = FRAMES) -> list[AgreementSample]:
    """Generate ``n`` agreement samples, drawing the frame for each example uniformly
    from ``frames`` (default: a mix of all three) so a generated set has real
    structural diversity, not just lexical substitution into one template."""
    rng = random.Random(seed)
    nouns = _filter_nouns(tokenizer)
    verbs = _filter_verbs(tokenizer, VERB_PAIRS)
    rc_verbs = _filter_verbs(tokenizer, RC_VERB_PAIRS) if "rc" in frames else []
    if len(nouns) < 4 or not verbs:
        raise RuntimeError(f"Need single-token nouns/verbs; got {len(nouns)} nouns, {len(verbs)} verbs.")
    if "rc" in frames and not rc_verbs:
        raise RuntimeError("Need at least one single-token RC verb pair for the 'rc' frame.")
    seen: set[str] = set(exclude or ())
    out: list[AgreementSample] = []
    attempts, max_attempts = 0, 400 * n + 2000
    while len(out) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could only build {len(out)}/{n} agreement samples.")
        frame = rng.choice(frames)
        v_s, v_p = rng.choice(verbs)
        # Which direction is "clean"? Flip per sample so the answer is a genuine mix of
        # is/are/was/were, not always the plural form -- earlier versions always built
        # "clean" as the plural-head sentence, so the model was only ever asked to produce
        # "are"/"were" and never learned the singular direction at all.
        plural_clean = rng.random() < 0.5

        if frame == "pp1":
            (h_s, h_p), (a_s, a_p) = rng.sample(nouns, 2)
            prep = rng.choice(PREPS)
            if plural_clean:
                clean_prompt, corr_prompt, answer = f"The {h_p} {prep} the {a_s}", f"The {h_s} {prep} the {a_p}", v_p
            else:
                clean_prompt, corr_prompt, answer = f"The {h_s} {prep} the {a_p}", f"The {h_p} {prep} the {a_s}", v_s

        elif frame == "pp2":
            (h_s, h_p), (a1_s, a1_p), (a2_s, a2_p) = rng.sample(nouns, 3)
            prep1, prep2 = rng.sample(PREPS, 2)
            if plural_clean:
                clean_prompt = f"The {h_p} {prep1} the {a1_s} {prep2} the {a2_s}"
                corr_prompt = f"The {h_s} {prep1} the {a1_p} {prep2} the {a2_p}"
                answer = v_p
            else:
                clean_prompt = f"The {h_s} {prep1} the {a1_p} {prep2} the {a2_p}"
                corr_prompt = f"The {h_p} {prep1} the {a1_s} {prep2} the {a2_s}"
                answer = v_s

        else:  # "rc": subject-relative-clause attractor
            (h_s, h_p), (a_s, a_p) = rng.sample(nouns, 2)
            vr_s, vr_p = rng.choice(rc_verbs)
            # NOTE: vr_p/vr_s already carry a leading space (needed so the verb tokenizes
            # correctly when used standalone as an answer elsewhere) -- no space after "that".
            if plural_clean:
                clean_prompt, corr_prompt, answer = f"The {h_p} that{vr_p} the {a_s}", f"The {h_s} that{vr_s} the {a_p}", v_p
            else:
                clean_prompt, corr_prompt, answer = f"The {h_s} that{vr_s} the {a_p}", f"The {h_p} that{vr_p} the {a_s}", v_s

        if clean_prompt in seen:
            continue
        if not _aligned(tokenizer, clean_prompt, corr_prompt):
            continue
        seen.add(clean_prompt)
        # Minimal-pair foil: the SAME verb in the opposite number (answer=v_p -> contrast=v_s
        # when the clean subject is plural, and vice versa). Linzen 2016 scores correctness as
        # P(answer) > P(contrast) at the verb slot, not argmax over the whole vocabulary.
        contrast = v_s if plural_clean else v_p
        out.append(AgreementSample(clean_prompt, corr_prompt, answer, frame, plural_clean, contrast))
    return out


def save_samples(samples, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def load_samples(path):
    with open(path) as f:
        return [AgreementSample(**json.loads(line)) for line in f if line.strip()]


class AgreementDataset(Dataset):
    """Single-token answer; accuracy uses argmax-VALID scoring (agreement_accuracy in
    ilm/eval/circuit.py), NOT exact-token match -- with a 49-pair verb pool, nothing in
    the prompt tells the model WHICH exact synonym a given example arbitrarily drew, so
    correctness is "predicted the right NUMBER" (any valid same-plurality verb), the same
    principle as greater-than's argmax-valid metric. ``valid_ids`` (all same-number verb
    token ids) is carried per example for that metric; the collator ignores extra keys."""

    def __init__(self, samples, tokenizer, max_length: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._valid_ids_cache = {
            plural: valid_verb_ids(tokenizer, plural) for plural in (True, False)
        }
        self.examples = [self._encode(s, tokenizer) for s in samples]

    def _encode(self, s, tok) -> dict:
        bos = bos_prefix(tok)
        prompt_ids = bos + tok(s.clean_prompt, add_special_tokens=False).input_ids
        ans_ids = tok(s.answer, add_special_tokens=False).input_ids   # one token
        clean_ids = (prompt_ids + ans_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + ans_ids)[: self.max_length]
        corr_ids = (bos + tok(s.corrupted_prompt, add_special_tokens=False).input_ids)[: self.max_length]
        # Linzen forced-choice ids: the correct verb token vs the same verb's wrong-number form.
        contrast_id = tok(s.contrast, add_special_tokens=False).input_ids[0] if s.contrast else -1
        return {
            "clean_ids": clean_ids,
            "labels": labels,
            "corr_ids": corr_ids,
            "clean_answer": s.answer,
            "valid_ids": self._valid_ids_cache[s.plural],
            "answer_id": ans_ids[0],
            "contrast_id": contrast_id,
            # Unified foil for the CONTRASTIVE attribution metric: the same-verb wrong-number
            # form, so attribution targets logit(correct)-logit(wrong-number) -- the Linzen
            # minimal-pair decision the eval scores. -1 (no foil) falls back to logp(answer).
            "foil_id": contrast_id,
            "prompt": s.clean_prompt,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
