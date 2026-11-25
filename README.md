# YoloMT: You Only Look Multiple Times

Code to train YoloMT end to end, from the paper "You Only Look Multiple Times: YoloMT, a Real-Time Streaming Model on
a T4".

YoloMT is a 449.8M-parameter streaming video captioner. Every second it writes a 30-50-word caption of the last 16 s
from the current frame and L cached past frames, with the lookback L chosen at run time from 0 to 16. Each frame goes
through the vision encoder once and is cached as 16 pooled tokens. The student is LiquidAI/LFM2.5-VL-450M plus a
1.05M-parameter pooler, distilled from Qwen3.8-27B in four stages. Weights: [Nessim/YoloMT](https://huggingface.co/Nessim/YoloMT).

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -e .            # captioning and training
pip install -e .[vllm]      # also labelling, deriving and GRPO, which run the teacher and the judges on vLLM
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
scripts/             labelling, derivation, calibration, the four training stages, captioning
data/                window lists, archives, fluency calibration
```

## Data and licenses

The code is under Apache-2.0 (`LICENSE`). The weights are under the LFM Open License v1.0 of LFM2.5-VL-450M. The
window lists hold video names only; the videos stay under their datasets' terms: LLaVA-Video-178K for academic
research and education, OpenVid-1M under CC BY 4.0, ShareGPT4Video (2 s labels only) under CC BY-NC 4.0, and each
under the terms of the sources its clips come from. Only captions written by the teacher are used for training.
