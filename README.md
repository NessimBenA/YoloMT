# YoloMT: You Only Look Multiple Times

Code to train and evaluate YoloMT end to end, from the paper "You Only Look Multiple Times: YoloMT, a Real-Time
Streaming Model on a T4".

YoloMT is a 449.8M-parameter streaming video captioner. Every second it writes a 30-50-word caption of the last 16 s
from the current frame and L cached past frames, with the lookback L chosen at run time from 0 to 16. Each frame goes
through the vision encoder once and is cached as 16 pooled tokens. The student is LiquidAI/LFM2.5-VL-450M plus a
1.05M-parameter pooler, distilled from Qwen3.8-27B in four stages. Weights: [Nessim/YoloMT](https://huggingface.co/Nessim/YoloMT).

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e .            # captioning and training
pip install -e .[vllm]      # also labelling, deriving, GRPO and evaluation, which run the teacher, the judges,
                            # the answerer and most baselines on vLLM
pip install -e .[bench]     # also rebuilding the benchmarks
pip install -e .[deploy]    # also the TensorRT engine
```

`requirements.txt` pins the versions the paper's model was trained with. Run every command from the repository root.

## Caption a video

```
python scripts/caption.py --weights Nessim/YoloMT --video clip.mp4 --lookback 16 --out clip.jsonl
```

Each line of `clip.jsonl` is `{"second": t, "caption": ...}`. `--weights` also takes a model folder written by
training, such as `runs/grpo/best`.

## Train end to end

`data/windows/` lists the windows the paper's model was trained on: per video, its archive, split and the end second
of each window. The videos come from LLaVA-Video-178K, OpenVid-1M and ShareGPT4Video on the Hugging Face Hub
(`data/archives.tsv`); `scripts/label.py` downloads each archive, decodes the listed videos and deletes it.

The paper's labels, steps 1 and 2 below, are on the Hub as the dataset
[Nessim/YoloMT-labels](https://huggingface.co/datasets/Nessim/YoloMT-labels): one folder per window list, text only.
To train from them, download them and decode their frames from the videos instead of running the teacher, then go
to step 3:

```
hf download Nessim/YoloMT-labels --repo-type dataset --local-dir labels
for s in captioner sft rl val; do python scripts/frames.py --labels labels/$s; done
```

They hold the same windows in the same order as the label folders of the paper's runs.

1. Teacher labels: Qwen3.8-27B-FP8 captions each window. The 2 s windows train stages 1 and 2; the 16 s windows
   train stage 3; the RL and validation windows also get 8 sampled captions for stage 4.

   ```
   python scripts/label.py --windows data/windows/captioner.tsv.gz --task 2s --out labels/captioner
   python scripts/label.py --windows data/windows/sft.tsv.gz --task 16s --out labels/sft
   python scripts/label.py --windows data/windows/rl.tsv.gz --task 16s --samples 8 --out labels/rl
   python scripts/label.py --windows data/windows/val.tsv.gz --task 16s --samples 8 --out labels/val
   ```

2. Claims and questions of the RL and validation windows, by Qwen3.5-9B:

   ```
   python scripts/derive.py --labels labels/rl labels/val
   ```

3. The reward's fluency scale. `data/fluency.json` is the paper's; this command recomputes it from your labels:

   ```
   python scripts/calibrate.py --labels labels/sft/llava_* --out data/fluency.json
   ```

4. Stage 1, the 2 s captioner: LoRA SFT reading one past frame and the previous caption.

   ```
   python scripts/train_2s.py --labels labels/captioner --out runs/captioner
   ```

5. Stages 2 and 3, sandwich SFT: each batch is trained at L = 0, a random middle L and the largest available L.
   First LoRA up to L = 5 on the 2 s labels, then every weight up to L = 16 on the 16 s labels.

   ```
   python scripts/train_sft.py --labels labels/captioner --init runs/captioner/best --out runs/sandwich \
       --k-max 5 --val-loops 0 1 2 3 4 5 --lr 2e-5 --max-new 48
   python scripts/train_sft.py --labels labels/sft --init runs/sandwich/best --out runs/sft --full \
       --k-max 16 --val-loops 0 1 2 4 8 16 --lr 1e-5 --vision-lr 2e-6 --max-new 80 --checkpoint-layers
   ```

6. Stage 4, GRPO with a KL penalty to the SFT model. The student trains on one GPU while the two judges, Qwen3.5-9B
   for the reward and phi-4 for checkpoint selection, take turns on another.

   ```
   CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py --labels labels/rl --val labels/val \
       --init runs/sft/best --out runs/grpo --judge-gpu 1
   ```

`runs/grpo/best` is the final model folder. Each script documents its method and settings in `--help`; the defaults
are the paper's. Every stage resumes from `last.pt` in its run folder when started again. `--micro` sets the windows
per forward pass, to fit the memory of your GPU.

## Evaluate

Every evaluation starts from the released weights (`--weights Nessim/YoloMT`, the default); every other model and
dataset is pinned to a revision.

### Benchmarks: caption, then answer

A describer captions each clip as a live stream, blind to the question; Mistral-Small-24B-Instruct-2501, which never
sees a frame, answers each multiple-choice question from the captions alone (`yolomt/bench.py`). The subsets are
OVO-Bench Real-Time (837 questions), MVBench without Fine-grained Pose (3,800), StreamingBench Real-Time (2,498) and
Video-MME short (900).

1. Rebuild the subsets from the Hugging Face Hub: questions, caption requests and the 1 fps, 256 x 256 frames. Each
   build asserts the digests of `data/bench.json` and fails if anything differs from the paper's.

   ```
   for b in ovo mvbench streaming videomme; do python scripts/bench_data.py --benchmark $b --out bench/$b; done
   ```

2. Caption every request at the protocol lookback L = min(t, 16), and at L = 0 for the gain from the past:

   ```
   for b in ovo mvbench streaming videomme; do
     for k in protocol 0; do
       python scripts/describe.py --model yolomt --bench bench/$b --loops $k --out captions/$b/yolomt-k$k.jsonl
     done
   done
   ```

   `--model` also takes the baselines of the paper's table and the teacher (`--help` lists them). FastVLM-0.5B,
   Mobile-VideoGPT-0.5B and TimeChat-Online-7B run in the environments of `requirements/`. `scripts/cut.py` cuts
   YoloMT's captions to a baseline's word counts, the paper's length control.

3. Answer, with the answerer split over `--tensor-parallel` GPUs, and report:

   ```
   python scripts/answer.py --bench bench --captions captions --names yolomt-kprotocol yolomt-k0 --floor \
       --out answers --tensor-parallel 2
   python scripts/bench_report.py --bench bench --answers answers --captions captions \
       --reference yolomt-kprotocol --out bench_report.json
   ```

   The report holds the accuracy on each benchmark and their mean, per task, the mean words per caption, and paired
   95% intervals (10,000 bootstrap resamples of the questions) of the reference minus every other answer set and of
   each describer's gain from L = 0.

### Held-out fact-F1

phi-4 scores a caption against the claims the teacher's sampled captions agree on (recall) and checks its atomic
facts against every claim they make (precision); fact-F1 is reported as a share of the teacher's own score, its
leave-one-out ceiling (`yolomt/heldout.py`). `data/windows/` lists the scored windows of LLaVA-Video test (`test`,
1,518 windows), held-out OpenVid (`openvid`, 1,008) and the LLaVA-Video dev split (`dev`, 1,103). Take their
labels from the labels dataset, decode their frames, then caption, judge and report:

```
hf download Nessim/YoloMT-labels --repo-type dataset --include "test/*" --local-dir labels
python scripts/frames.py --labels labels/test
python scripts/describe.py --model yolomt --windows data/windows/test.tsv.gz --labels labels/test --loops 0 16 \
    --out captions/test/yolomt.jsonl
python scripts/heldout_judge.py --labels labels/test --windows data/windows/test.tsv.gz \
    --captions captions/test/yolomt.jsonl --out judged/test
python scripts/heldout_report.py --judged judged/test --captions captions/test/yolomt.jsonl
```

The same commands with `openvid` in place of `test` score held-out OpenVid. The evidence controls read, at
L = 16, the current frame repeated, another video's past, or the window's own past shuffled, on the dev windows
(labels and frames in `labels/dev` as above):

```
python scripts/controls.py --weights Nessim/YoloMT --labels labels/dev --windows data/windows/dev.tsv.gz \
    --out captions/controls
python scripts/heldout_judge.py --labels labels/dev --windows data/windows/dev.tsv.gz \
    --captions captions/controls/*.jsonl --out judged/controls
python scripts/heldout_report.py --judged judged/controls --reference plain --controls
```

With these labels the scored windows, targets and teacher ceilings are the paper's. Labelling a list afresh
instead (`scripts/label.py --samples 8`, then `scripts/derive.py`) moves the ceilings slightly, since the teacher
samples at temperature 0.7, and leaves out a listed window that gets no ceiling.

### T4 timing

Eager PyTorch, FP16, batch 1: the median time of a caption forced to 50 new tokens at L = 0 and 16, and the process
memory:

```
python scripts/latency.py --model yolomt --out latency/yolomt.json
python scripts/latency.py --model lfm25-vl-450m --out latency/lfm25-vl-450m.json
```

The real-time engine: build the FP16 TensorRT engines from the released weights (checked against float32 PyTorch at
build time), time the 30 FPS schedule on three public clips, and serve paced streams:

```
python deploy/engine.py --weights Nessim/YoloMT --out engines
python deploy/run.py timing --engines engines --out timing.json
python deploy/run.py serve --engines engines --streams 3 --loops 16 --out streams
```

The paper's times are those of a T4; another GPU gives other times.

### Reproduced from the released weights

Rerun with this code from `Nessim/YoloMT`, captions, answers and judgments all recomputed; the paper's values in
parentheses:

| | OVO-RT | MVBench | StrBench | VMME | mean |
|---|---|---|---|---|---|
| YoloMT, L = min(t, 16) | 46.8 (45.9) | 39.8 (39.5) | 50.0 (50.2) | 47.2 (46.2) | 45.9 (45.5) |
| YoloMT, L = 0 | 47.4 (47.2) | 36.6 (36.4) | 48.0 (48.6) | 40.1 (40.1) | 43.0 (43.1) |
| no captions | 27.8 (27.8) | 27.2 (27.6) | 33.4 (33.5) | 28.7 (28.7) | 29.3 (29.4) |

The gain from L = 0 is +2.9 [+1.6, +4.2] points (+2.5 [+1.2, +3.7]). Held-out fact-F1, % of the ceiling: OpenVid
84.0 (83.8) at L = 0 and 92.6 (92.6) at L = 16, test 53.0 (53.3) and 72.1 (71.8). The controls on dev, at L = 16:
own past 71.5 (72.2; 52.3 (52.7) at L = 0), current frame repeated 52.1 (52.2), another video's past 19.3 (18.7),
own past shuffled 70.8 (71.0). Greedy bf16 decoding differs across
GPUs, so about half of the captions differ in wording from the paper's; the paper measures that re-captioning moves
the benchmark mean by up to 0.64 points. From the paper's stored captions and answers, the code reproduces every
accuracy, interval and share of the paper exactly.

Not covered: the ablations, which train other models, and the quantized rows of the table, whose checkpoints are not
published.

## Layout

```
yolomt/model.py      the student: LFM2.5-VL-450M with the pooler, prompts at lookback L, loss, decoding, model folders
yolomt/data.py       labelled windows, their frames and past frames
yolomt/video.py      archives, video ids, frame decoding
yolomt/teacher.py    what the teacher sees and is asked
yolomt/facts.py      claims and questions, and scoring a caption against them
yolomt/judge.py      the vLLM judges
yolomt/reward.py     the GRPO reward
yolomt/schedule.py   learning-rate schedule and exit rule
yolomt/bench.py      the benchmark protocol: questions, caption requests, the answerer's prompt and parser
yolomt/benchmarks.py the four benchmark subsets rebuilt from the Hub
yolomt/describers.py YoloMT, the baselines and the teacher as captioners
yolomt/answer.py     the answerer and the caption sets
yolomt/stats.py      accuracy and paired bootstrap intervals
yolomt/report.py     the benchmark report
yolomt/heldout.py    held-out fact-F1: ceilings, shares, intervals, controls
yolomt/latency.py    eager timing
yolomt/graphs.py     the engine's graphs, exported to ONNX
scripts/             labelling, derivation, calibration, the four training stages, captioning, evaluation
deploy/              the TensorRT engine and its 30 FPS runtime
data/                window lists, archives, fluency calibration, benchmark digests, the engine's vocabulary
requirements/        the environments of three baselines
```

## Data and licenses

The code is under Apache-2.0 (`LICENSE`). The weights are under the LFM Open License v1.0 of LFM2.5-VL-450M. The
window lists hold video names only; the videos stay under their datasets' terms: LLaVA-Video-178K for academic
research and education, OpenVid-1M under CC BY 4.0, ShareGPT4Video (2 s labels only) under CC BY-NC 4.0, and each
under the terms of the sources its clips come from. Only captions written by the teacher are used for training.
The labels dataset is under CC BY-NC 4.0, and its use must also follow the terms of
the source videos. The benchmarks are rebuilt from their own repositories and stay under their terms;
`data/bench.json` holds digests, no question, option or frame.
