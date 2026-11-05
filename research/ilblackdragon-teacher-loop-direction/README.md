# Teacher-guided direct removal of a repetitive-reasoning direction

**Outcome: stopped at diagnosis; no supported weight edit.** The single eligible
teacher query also exhausted the fixed reasoning budget without a final answer.
The predeclared stop condition was followed instead of changing the protocol or
manufacturing a successful contrast. No candidate, patch or submission exists.

## Hypothesis and access

Illia Polosukhin proposed using a larger Qwen model's behavior to improve the
smaller model and requested a direct weight edit inspired by
[weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/).
Codex implements this bounded diagnostic experiment without gradient training.

The hypothesis is that a direction associated with repeated student reasoning
can be identified by contrasting that reasoning with a correct teacher answer,
then partially removed from selected student MLP output columns. Teacher text is
replayed through the student: neither weights nor activations are subtracted
between models of different sizes. This is a candidate causal intervention, not
proof that a contrast direction uniquely represents repetition.

Only public checkpoints, public repository files, and the downloaded public
validation manifest and its public upstream dataset selections are used. No
League private examples are accessed. The upstream GSM8K split called `test` is
public data explicitly named in that public manifest, not the League private test.

Layers 15 and 19 are chosen from the post's prior result; they are not independently
discovered here. This experiment does not reproduce the authors' released diff.

## Parent and evaluation

- Competition: `qwen35-08b-v1`.
- Original parent: `0ea62f40-7cd8-4d98-9ebb-484192fa05eb`.
- Parent content hash: `708085f8b44fa704601967f3f771b8ec133d9c91dc4f49bd8c7686bd47d537b8`.
- Pinned evaluation: `public-ladder500-8k-v2`.
- Public validation SHA-256: `996573d687b0d0f1e088f1e918cd153fa0f80cff454a7e9a7ab5ac2eb374d4d2`.
- Teacher: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Public GSM8K revision: `740312add88f781978c0658806c59bc2815b9866`.

The original checkpoint and `delta-model.json` remain unchanged. The local
workspace reuses the already downloaded model and teacher via directory symlinks;
no symlinks or model artifacts are included in this research PR.

## Fixed design

Shuffle the 64 public GSM8K manifest positions with Python `random.Random(42)`;
use the first four for development and next four for holdout. Verify every
selected document's SHA-256 over sorted compact UTF-8 JSON before inference.
Freeze the split before model execution. No holdout trace influences selection
of teacher queries, directions, columns, or edit strength.

Each prompt requests step-by-step reasoning and a boxed final answer. Use each
model's chat template with thinking enabled, no few-shot examples, seed 1000,
temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5, and at most 1,536 new
tokens. Presence penalty applies only to generated tokens. Student baseline and
candidate run in float32; the teacher runs in BF16 on the local AVX512-BF16 CPU.

This is a local diagnostic proxy, not the official five-shot GSM8K metric or
8,192-token evaluation. Correctness uses the CLI's public math scorer only after
the generated reasoning has a closing `</think>` tag. A truncated reasoning trace
without a final answer is scored zero, even if a correct number appears inside
the reasoning. Completion, truncation, repetition, and correctness are reported
separately; a token-limit hit alone is not called a loop.

The prespecified repetition detector finds generated 8-token sequences occurring
at least three times. It marks their token positions, then flags a loop when at
least 20% of the final 256 generated tokens are marked. Detected development
traces must also be inspected to distinguish genuine repetition from legitimate
repeated arithmetic. Query the teacher on the first two incorrect flagged
development examples, at most. Only a correct, nonrepetitive teacher answer
paired with an inspected repetitive incorrect student answer supports an edit.
If no such pair is available, stop and report no supported edit.

For supported pairs, replay the student's trace and the teacher's reasoning
(before `</think>`) through the original student. At each target layer, contrast
the mean down-projection output on the student's flagged tail tokens with the
mean output on the teacher reasoning's final 256 tokens. Average contrasts
equally across paired examples and normalize to direction `u`. Rank columns by
`abs(u @ W) * RMS(loop input activation)` and select 64 columns. Apply:

```text
delta_selected = -0.5 * outer(u, u @ W_selected)
scale = min(1, 0.005 * ||W||_F / ||delta_selected||_F)
W_selected_new = BF16(W_selected_float32 + scale * delta_selected)
```

Only the two down-projection matrices may change. There is one fixed candidate,
no gradient updates, no optimizer, no strength search, and no retuning from
holdout results. BF16 rounding can slightly change the realized norm and rank.
The teacher/student contrast can capture topic, position, or style differences
as well as repetition; paired examples reduce but do not eliminate that confound.

## Reproduction and budget

Dependencies are in `requirements.lock.txt`. Python 3.12, Linux, Ryzen 9 9950X3D,
at most 16 CPU compute threads, 64 GiB address-space limit per inference process.
Stages run sequentially. The approved budget is two hours and 5 GiB additional
disk, reusing cached teacher weights; no rented compute or automatic submission.
`plan.json` records the fixed design and approval. A shared absolute deadline is
enforced by a process alarm and generation-time checks. Raw traces stay under
the local workspace's `local/` directory and must not be committed.

Initialize a new workspace with the exact parent and ensure its evaluation/hash
match the pinned identifiers above. Reuse the pinned teacher snapshot, or download
it separately under an appropriate download/storage budget. Set `plan.json`'s
`started_at_unix` and `deadline_unix` to the authorized start and start plus 7200.

```bash
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage baseline
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage teacher
# Inspect flagged development traces. If there are valid pairs, write
# local/diagnosis.json with confirmed_loop_ids before constructing an edit.
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage edit
# Only if an edit was supported and edited.safetensors exists:
python research/ilblackdragon-teacher-loop-direction/run.py --workspace WORKSPACE --stage candidate
cd WORKSPACE
delta patch build --parent ./model --modified ./edited.safetensors --output ./submission
delta validate ./submission
```

`report.py --workspace WORKSPACE` exports aggregate results and IDs without text
or token arrays. `audit.py --workspace WORKSPACE` checks parent file hashes; if
a candidate exists it also checks names/shapes/dtypes, finiteness, target scope,
and bitwise equality of patch reconstruction with `edited.safetensors`.

## Interpretation and submission

The completed run produced:

| Local diagnostic group | Problems | Correct final answers | Hit 1,536-token cap | Strict repetition flags |
| --- | ---: | ---: | ---: | ---: |
| Original development | 4 | 0 | 4 | 1 |
| Original holdout | 4 | 0 | 4 | 0 |
| Teacher on eligible development problem | 1 | 0 | 1 | 0 |

There are no candidate scores; they are null, not zero. The exact public question
IDs, document hashes, settings, aggregates and per-example metric summaries are
in `results.json`. Raw traces are kept outside Git.

On public problem 421, both models computed 12 cats, then repeatedly reconsidered
whether chasing a mouse meant leaving the rock. Neither closed its reasoning or
gave a final answer before the cap. The student's exact-tail repetition coverage
was 29.30%; the teacher's was 10.16%. A low exact-repetition score therefore did
not guarantee progress. Development problems 438 and 170 also showed repeated
reconsideration with varied phrasing but did not qualify under the fixed detector.
These observations are recorded in `diagnosis.json`; the threshold was not tuned.

The larger teacher was not a successful contrast for this particular failure
under these settings. The direct-edit routine in `run.py` is prepared but was
**not exercised**, and its efficacy remains untested. This is not evidence that
direct weight editing cannot work. It also does not establish that the models
would fail with the official longer token allowance or a different prompt.

Baseline generation took 1,708.41 seconds; teacher generation took 501.33 seconds.
Including loading, the baseline and teacher stages took 1,709.83 and 502.31
seconds. End-to-end wall time including preparation and the requested source
review was about 41 minutes, within the two-hour ceiling. Peak measured RSS was
14.95 GiB for the BF16 teacher. All seven original downloaded model files passed
their size/hash checks. No weight changed, no patch was built or submitted, and
no evaluation quota was consumed. The remaining budget was not used to start a
second candidate search or change the predeclared stopping rule.

Four public holdout examples cannot establish general capability improvement.
The short token cap can prevent answers that would complete under the official
budget, or miss loops that emerge later. Sampling and CPU/precision differences
also prevent interpreting these local figures as official scores.

No submission is authorized by the local-compute approval. Any resulting patch
must first be documented in this research PR and separately approved for quota
consumption. Candidate official scores remain unavailable until a submitted
evaluation completes. Negative or inconclusive diagnoses are retained.

For context only, the existing original League result on `public-ladder500-8k-v2`
is 41.830071900430084 public / 18.6 private. Those platform-verified scores are
different measurements from this short-budget local diagnostic. There is no new
official score or pending evaluation from this run.

## Additional references requested during execution

The user supplied [Zenodo record 23958070](https://zenodo.org/records/23958070)
and [Boris Knyazev's post](https://x.com/borisaknyazev/status/2118065460377792918)
while the original-model baseline was running. `references.json` records source
identity, retrieval details, hashes, methodological caveats, and possible future
experiments. These sources did not change the fixed design of this run. The
third-party documents, media, and code are not redistributed in this PR.
