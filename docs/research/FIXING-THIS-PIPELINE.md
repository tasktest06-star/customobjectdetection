# Fixing this pipeline

An ordered remediation guide for the code in this repository. The
[gap analysis](01-gap-analysis.md) says what is wrong and where; this says what
to do about it, in the order that matters, with code that has been run.

**Read this first if you are about to change the code.** Read
[00-executive-summary.md](00-executive-summary.md) first if you are deciding
whether the approach is right at all.

## The one-paragraph version

Three defects make every number this pipeline reports untrustworthy, and they
are all cheap to fix. The reported metric measures agreement with the teacher
rather than correctness. The train and validation split leaks, and moves under
you between rounds. And the self-training agreement filter keeps everything it
disagrees with, so it filters only where filtering was least needed. Fix those
three before tuning anything, because until then you cannot tell whether a
change helped.

## Order of work

| # | Fix | Effort | Why this order |
|---|---|---|---|
| 1 | Grouped, seeded split | ~20 lines | Until this is right, every comparison is meaningless |
| 2 | A hand-annotated gold set | Hours of annotation | Until this exists, the metric cannot detect failure |
| 3 | Remove the keep-everything fallback | ~3 lines | Currently defeats the filter it belongs to |
| 4 | Cache the teacher pass | ~10 lines | Pure waste: the most expensive step repeats identically |
| 5 | Best-match label resolution with false-friend guards | ~30 lines | Silently cross-assigns confusable classes |
| 6 | Per-clip prompt restriction | moderate | Cost and precision both |
| 7 | Add a licence file, or swap the copyleft dependency | small | Decide before anyone depends on the output |
| 8 | Wire or delete the dead config keys | small | A key read nowhere invites tuning that cannot work |

## 1. The split, and it is the first thing to do

`src/annotation/coco_builder.py` lines 97 to 99:

```python
random.shuffle(detections)
n_train = int(len(detections) * self.train_ratio)
splits = {"train": detections[:n_train], "val": detections[n_train:]}
```

Two problems. The unit of shuffling is a **frame**, and frames sampled from one
video are near duplicates, so the same scene and the same physical instrument
land on both sides. And there is no seed, while `build_dataset` is called once
per self-training round, so each round gets a different split and round-over-
round metrics are computed on different data.

Replace with this. Filenames already embed the video stem, from
`src/video_pipeline/frame_extractor.py` line 51, so the grouping key is
available with no new metadata.

```python
import random
import re
from pathlib import Path


def video_id_from_path(image_path):
    """'<video_stem>_<frame_idx>.jpg' -> '<video_stem>'."""
    stem = Path(image_path).stem
    m = re.match(r"^(.+)_(\d{4,})$", stem)
    return m.group(1) if m else stem


def grouped_split(detections, train_ratio=0.8, seed=0,
                  group_of=video_id_from_path):
    """Allocate whole videos to train or val. Never splits a video.

    Largest group first, each going to whichever side is furthest below its
    quota. That keeps the ratio close even when video lengths differ wildly,
    which a simple sequential fill does not.

    Seeded, so the split is reproducible and its variance measurable. Raises
    below two groups, because a frame-level split there would leak.
    """
    groups = {}
    for det in detections:
        groups.setdefault(group_of(det["image_path"]), []).append(det)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError(
            "found {} source video(s); a grouped split needs at least 2. "
            "A frame-level split would leak.".format(len(names)))

    rng = random.Random(seed)
    rng.shuffle(names)                       # the seed breaks ties below
    names.sort(key=lambda n: -len(groups[n]))

    total = len(detections)
    quota = {"train": train_ratio * total, "val": (1.0 - train_ratio) * total}
    bucket = {"train": [], "val": []}
    for name in names:
        side = max(bucket, key=lambda s: quota[s] - len(bucket[s]))
        bucket[side].extend(groups[name])
    if not bucket["val"] or not bucket["train"]:
        raise ValueError(
            "could not form a non-empty split from {} videos at ratio {}"
            .format(len(names), train_ratio))
    return bucket["train"], bucket["val"]
```

Verified behaviour, run before publishing: no video appears on both sides at
any seed, every frame is placed, both sides are non-empty, and the result is
deterministic given a seed. With five equal videos it hits 0.80 exactly. With
very uneven lengths it lands between 0.88 and 0.97, because whole videos cannot
be divided and that is the honest cost of not leaking.

**Group by recording session if you can, not merely by video.** Several videos
shot in one room on one day share background and lighting, so a video-level
split still leaks some of that. A session identifier in the manifest is worth
more than any modelling change in this list.

**Then publish the gap.** Run the evaluation once grouped and once ungrouped
and report the difference as your leakage estimate. Expect the ungrouped number
to be higher by a lot. That difference is how much every previously reported
figure was inflated.

## 2. The gold set, because the metric currently cannot fail

`scripts/evaluate.py` reports mean average precision on the validation split,
but that split is pseudo-labelled by the same teacher that produced the
training labels. The number measures how well the student reproduces the
teacher, including the teacher's mistakes. It rises as the student conforms and
it cannot fall when the teacher is wrong.

There is no code fix. You need ground truth that never passed through the
pipeline.

- **Three to four frames per clip, across every clip.** Not many frames from a few clips: at equal cost, spreading gives more than five times the effective sample size, and dense annotation of a few clips is the worst available use of the budget.
- **Record which classes you actually checked on each frame.** A class you did not look for is unknown, not absent. Scoring it as a miss inflates precision.
- **Budget roughly 26 seconds per box end to end**, not the four seconds often quoted, which is mouse-gesture time on familiar classes. Verifying machine proposals is about 1.5 times faster than drawing from scratch, not the ten to twenty times sometimes claimed.
- **Group the evaluation by recording session** and report an interval, not a point. With few sessions the interval is wide and that is the truth rather than a presentation problem.

## 3. The agreement filter is close to a no-op

`src/training/self_trainer.py` lines 85 to 87 and line 96:

```python
if sd is None or len(sd["boxes_xyxy"]) == 0:
    ensembled.append(td)          # keeps ALL teacher boxes
    continue
...
idx = kept if kept else list(range(len(td["boxes_xyxy"])))   # again
```

The docstring calls this preserving coverage on hard examples. The effect
inverts the intent: on easy images where the student agrees, boxes are filtered;
on hard images where it disagrees, **everything** is kept unfiltered. So the
filter removes boxes only where it was least needed and preserves unverified
output exactly where verification was the point.

```python
            if sd is None or len(sd["boxes_xyxy"]) == 0:
                continue                      # contribute nothing
            ...
            if not kept:
                continue                      # contribute nothing
            idx = kept
```

There is a deeper problem the fix does not address. The student was trained on
the teacher's labels, so student-teacher agreement is not independent evidence;
it measures how well distillation worked. Real agreement needs two
**independently pretrained** detectors with different training mixes. The
repository already has two, Grounding DINO and OWLv2, but
`pseudo_labeling.labeler` selects one. Run both and keep detections where they
agree on class and overlap.

## 4. The teacher pass repeats identically every round

`src/training/self_trainer.py` lines 136 and 139 relabel the same images with
the same prompts and thresholds every round. The model is in evaluation mode, so
the output is identical each time. Only the student changes.

So rounds beyond the first add no new teacher information. They re-filter one
fixed teacher output against a progressively more teacher-conformant student,
which is exactly the diminishing returns the README reports, and it is
structural rather than empirical. It is also the most expensive operation in the
pipeline, repeated for nothing.

Compute it once, cache it, and reuse. If rounds are to add information,
something has to change between them: the prompts, the thresholds, the frames,
or a second independent teacher.

## 5. Label resolution cross-assigns confusable classes

`src/annotation/coco_builder.py` lines 35 to 41 test substring containment in
both directions and return the **first** match, so resolution depends on the
order classes were declared. With a fine-grained vocabulary, `test tube holder`
resolves to `test_tube` if that is listed first, and the fragment `tube`
resolves to `test_tube` as well.

This matters more than it looks, because `grounding_dino.py` line 55 returns the
detector's text spans rather than clean class names, and with a
period-separated prompt those spans can be fragments or merged across class
boundaries.

```python
def match_label(raw, class_names, max_distance=2, min_substring=4):
    """Best match, not first. Refuses known false friends. Raises on a tie."""
    low = " ".join(str(raw).lower().replace("_", " ").split())
    norm = {c: " ".join(c.lower().replace("_", " ").split()) for c in class_names}

    for c, n in norm.items():                     # 1. exact
        if low == n:
            return c

    scored = [(levenshtein(low, n), c) for c, n in norm.items()]
    scored = [(d, c) for d, c in scored if d <= max_distance]
    if scored:                                    # 2. NEAREST, not first
        best = min(d for d, _ in scored)
        winners = sorted({c for d, c in scored if d == best})
        if len(winners) > 1:
            raise ValueError("{!r} is equidistant from {}".format(raw, winners))
        return winners[0]

    cands = []                                    # 3. guarded substring
    for c, n in norm.items():
        if any(bad in low for bad in FORBIDDEN.get(c, ())):
            continue
        if len(n) >= min_substring and n in low:
            cands.append((len(n), c))
    if cands:
        longest = max(n for n, _ in cands)
        winners = sorted({c for n, c in cands if n == longest})
        if len(winners) > 1:
            raise ValueError("{!r} matches {} equally".format(raw, winners))
        return winners[0]
    return None
```

`FORBIDDEN` is where the real content lives, and it is not optional. Verified
by reading the actual dataset category files: in one major dataset the fume hood
entry is a kitchen range hood, and it is a **frequent** class, so a detector is
confidently trained on the wrong object under exactly the name you would prompt
with. The shaker entry is a condiment shaker.

```python
FORBIDDEN = {
    "test_tube": ("holder", "rack", "stand"),
    "fume_hood": ("kitchen", "range", "exhaust", "cooker", "stove"),
    "orbital_shaker": ("condiment", "salt", "pepper", "cocktail"),
    "analytical_balance": ("bathroom", "kitchen", "weighing"),
    "conical_flask": ("thermos", "vacuum", "hip"),
    "beaker": ("coffee", "mug"),
}
```

Verified: `test tube holder` resolves correctly, `kitchen range hood`,
`condiment shaker` and `bathroom scale` all resolve to nothing, and a typo like
`centrifug` still reaches `centrifuge`.

Also count the labels that resolve to nothing instead of dropping them at line
60. A high unmatched rate means the prompts need rewriting, and that is worth
knowing before training on the result.

## 6. Prompt each clip with only its own vocabulary

`src/pseudo_labeling/grounding_dino.py` line 38 joins every class into one
prompt. Three consequences.

The text encoder has a bounded input, so past that bound the prompt truncates
and classes near the end are simply never detected, with no warning. A growing
vocabulary therefore degrades invisibly.

Cost is roughly linear in concepts queried. Restricting to the classes a clip is
actually labelled with cuts it proportionally.

Precision improves for free, because most false positives become impossible.

Then go further: also prompt three to five **confusable classes the clip is not
labelled with**. Every detection of those is guaranteed wrong, so it becomes an
explicitly labelled negative at no extra cost.

## 7. Decide the licence question before anyone depends on the output

`requirements.txt` pins a copyleft detector and the repository has no licence
file at all. Every published laboratory-equipment baseline uses the same
encumbered family, so their numbers are citable but their weights are not usable
in a non-copyleft project.

If the work is research only, this does not bind and you can ignore it. If
anything ships, swap to a permissively licensed detector. The
`apache2-rtdetr-pipeline` branch here already does that, and its detector
wrapper handles the two fiddly parts correctly: the one-indexed annotation to
zero-indexed model offset, and emitting well-formed empty targets for
unannotated images so the matching loss stays valid.

Note that the other side of this is not as clean as it looks either. The
pretraining corpus behind the open-vocabulary detectors states academic use
only, which transitively affects those weights. See
[02-datasets-and-licences.md](02-datasets-and-licences.md).

## 8. Dead configuration keys

`self_training.confidence_threshold` is set in the config and documented in the
README, but a search across every Python file finds no reader. The student
inference threshold is a hardcoded default. They coincide today, so there is no
visible symptom, and tuning the config silently does nothing.

Separately, `self_trainer.py` line 131 passes a hardcoded epoch count, ignoring
`training.epochs`, which is honoured only on the non-self-training path. So
enabling self-training silently changes the training length. `agree_iou` is
reachable only as a default and is absent from the config entirely.

Wire them through or delete them. A key read nowhere is worse than a missing
one, because it invites tuning that cannot work.

## What not to bother with

From the research, with the evidence:

- **Do not build active learning.** At this budget, reported results against random selection are mostly negative, and you sit below the smallest budget at which any method has beaten random for detection.
- **Do not rely on propagation for training labels.** The one clean ablation found tracker-mined boxes contributed 0.95 of an 11.90 gain, and in a second experiment tracker-only pseudo-labels were net negative. Use propagation to interpolate, not to generate.
- **Do not expect self-training to transform anything.** Measured gains are a few points, and the largest single factor in the reference work was initialisation rather than the self-training loop.

## Where the implemented version lives

Everything above, plus the gold set tooling, the statistics, the exporters and a
dependency-free test suite, is implemented in
`tasktest06-star/VLM_tunning`. Its `docs/GETTING_STARTED.md` runs the whole
pipeline in ten minutes with nothing installed, and `docs/MODULES.md` is the
module reference. If you are deciding whether to patch this pipeline or start
from that one, the honest answer is that the architecture here is sound and the
problems are in the details, so patching is reasonable; but the gold set and
the statistics do not exist here at all and that is most of the work.
