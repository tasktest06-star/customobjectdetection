# Research: object recognition in video from limited video-level labels

Research conducted 2026-10-04 on recognising laboratory equipment in video when the only
supervision is a handful of clip-level labels. The question that prompted it was how to fine-tune a
vision-language model under that constraint. The answer turned out to be that you should not, and
these documents set out why and what to do instead.

## Start here

| Document | Read it for |
|---|---|
| [00-executive-summary.md](00-executive-summary.md) | The answer. Verified facts, the recommended pipeline, the honest ceiling, and a decision guide. Stands alone. |
| [01-gap-analysis.md](01-gap-analysis.md) | Twelve defects in this repository's pipeline, each with a file and line number and a fix. Most immediately actionable. |
| [02-datasets-and-licences.md](02-datasets-and-licences.md) | Reference tables. Public data you can download, which detector has seen which class, the homonym trap, and the licence landmines. |
| [03-tracking-and-open-vocabulary.md](03-tracking-and-open-vocabulary.md) | Full study. Trackers, adding a class without retraining, and evaluating with no box ground truth. |
| [04-unlabelled-pool-and-annotation.md](04-unlabelled-pool-and-annotation.md) | Full study. Semi-supervised detection from zero boxes, pseudo-label constraints, and the active-learning verdict. |
| [05-annotation-budget-verification.md](05-annotation-budget-verification.md) | **Supersedes the annotation guidance in 03 and 04.** Refutes the headline one-box-per-clip claim, resolves the gold-set size, and gives the hour-by-hour allocation. |

If you have five minutes, read the executive summary. If you are about to change code, read the gap
analysis. If you are about to spend your own hours annotating, read document 05 first, because it
overturns the advice the earlier studies gave. Documents 03 and 04 are long primary research and are
meant to be searched rather than read front to back.

## The three findings that matter most

**You are not starting from zero box labels.** Roughly 22,000 CC BY 4.0 box-annotated
laboratory-apparatus instances are downloadable across four datasets, two of them video-derived.
The gap is specific and should direct labelling effort: no public boxes exist anywhere for
centrifuge, autoclave, orbital shaker, spectrophotometer, or a real fume hood.

**The current evaluation cannot detect failure.** The validation split is pseudo-labelled by the
same teacher that produced the training labels, so the reported metric measures agreement with the
teacher rather than correctness. It will report success while the model is confidently wrong. A
small hand-annotated gold set is the prerequisite for trusting anything else in the project.

**Fine-tuning an open-vocabulary detector destroys the property a growing class list depends on.**
Reported collapse is from 51.90 to 0.10 average precision at 0.5 after a single-class fine-tune,
against 51.90 held exactly by a frozen detector with visual prompting. The magnitude is under
verification; the direction is not in question.

## How to read the confidence markers

Numbers in these documents are tagged as measured in a source, vendor-claimed, or estimated. Treat
them as what the source reports, not as what this project will reproduce. Several cited benchmarks
are close to saturated and were collected under far easier conditions than handheld laboratory
footage.

Documents 03 and 04 are primary research that has **not** completed adversarial verification. Their
headers record the known uncertainties. Documents 00, 01 and 02 reflect later checking and win
where they disagree with 03 or 04.

## Known open questions

Two that appeared here originally are now resolved, in document 05.

- **Resolved. The gold set is 450 frames**, as 3 frames per clip across all 150 clips, scored by 5-fold cross-validation over clips rather than a fixed holdout. Documents 03 and 04 disagreed two-fold and were both arguing about a saturating axis; the binding constraint is clip count, not frame count.
- **Resolved and refuted. The one-box-per-clip claim was an arithmetic artefact.** The reported 58% and 88% turned out to be the same result stated two ways, and the unit was per image rather than per clip. Verify-and-correct remains the top recommendation, but on different evidence.
- **Still open.** The catastrophic-forgetting magnitude came from a single-class fine-tune that may be degenerate. Whether it generalises to a realistic multi-class fine-tune is under verification.
- **Still open.** A claimed fivefold CUDA graphs speedup traces to one unresolved issue thread and drives the best-case cost model. Plan with the slower number.
- **Newly open.** The performance prior in the executive summary is probably too optimistic. Zero-shot median average precision on a 35-domain suite is 11.9 to 18.4, not the COCO figures.

## Method

Nine general technique families and six setup-specific routes were surveyed, each followed by an
adversarial verification pass whose instruction was to refute rather than confirm. Two gap studies
covered tracking and the unlabelled pool. A six-cluster attack on those studies and a dedicated
check of the annotation economics were still running when this was committed, and corrections will
follow rather than being left to stand.

Dataset category claims were verified by downloading and searching the actual category files.
Licences were read from the repositories rather than from summaries. SAM 3's existence, model
identifiers, gating and licence terms were confirmed directly against arXiv and Hugging Face.
