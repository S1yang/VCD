# chair_caption_llava15.py
# useage:
# cd ~/VCD
# export CUDA_VISIBLE_DEVICES=0

# python experiments/eval/chair_caption_llava15.py \
#   --model-path liuhaotian/llava-v1.5-7b \
#   --image-folder /xxx/val2014 \
#   --image-list /xxx/random_sellect_500_hallu_imgs.txt \
#   --limit 500 \
#   --out-jsonl experiments/output \
#   --use-vcd \
#   --noise-step 500 \
#   --cd-alpha 1 \
#   --cd-beta 0.1 \
#   --max-new-tokens 256

import argparse
import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import Optional
import torch
from transformers import set_seed

# ===== make sure we import the VCD repo's llava (experiments/llava) =====
import sys
THIS = Path(__file__).resolve()
VCD_ROOT = THIS.parents[2]              # ~/VCD
EXP_ROOT = VCD_ROOT / "experiments"
sys.path.insert(0, str(EXP_ROOT))       # so "import llava" uses ~/VCD/experiments/llava

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
)

# ===== VCD utils (repo root has vcd_utils/) =====
sys.path.insert(0, str(VCD_ROOT))
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling


def coco_image_id_from_name(name: str) -> int:
    """
    COCO_val2014_000000239837.jpg -> 239837
    000000239837.jpg              -> 239837
    """
    stem = Path(name).stem
    m = re.search(r"(\d{6,12})$", stem)
    if not m:
        raise ValueError(f"Cannot parse image_id from filename: {name}")
    return int(m.group(1))


def read_image_list(
    image_folder: Path,
    image_list_txt: Optional[Path],
    limit: Optional[int]
):
    if image_list_txt is not None:
        lines = [x.strip() for x in image_list_txt.read_text(encoding="utf-8").splitlines() if x.strip()]
        paths = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = image_folder / ln
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            paths.append(p)
    else:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        paths = []
        for e in exts:
            paths += sorted(image_folder.glob(e))
        if not paths:
            raise FileNotFoundError(f"No images found in {image_folder}")

    if limit is not None:
        paths = paths[:limit]
    return paths


def main():
    parser = argparse.ArgumentParser()

    # model / data
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--image-list", type=str, default=None,
                        help="可选：txt，每行一个图片文件名/路径；不提供则扫描 image-folder")
    parser.add_argument("--limit", type=int, default=500)

    # output
    parser.add_argument("--out-jsonl", type=str, required=True)

    # prompt / conv
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--prompt", type=str, default="Please describe this image in detail.")

    # generation
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    # vcd
    parser.add_argument("--use-vcd", action="store_true", default=True)
    parser.add_argument("--noise-step", type=int, default=500)
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    disable_torch_init()

    # enable VCD sampling patch (so generate can take images_cd/cd_alpha/cd_beta)
    if args.use_vcd:
        evolve_vcd_sampling()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model.eval()

    image_folder = Path(args.image_folder)
    image_list_txt = Path(args.image_list) if args.image_list else None
    img_paths = read_image_list(image_folder, image_list_txt, args.limit)

    # ===== auto build output filename from args (only change output naming) =====
    out_arg = Path(args.out_jsonl)

    # 允许你传一个目录：--out-jsonl experiments/output
    # 或者传一个前缀路径：--out-jsonl experiments/output/run
    # 或者传一个完整文件名（仍支持）
    if out_arg.suffix.lower() == ".jsonl":
        out_dir = out_arg.parent
        user_prefix = out_arg.stem  # e.g. "[VCD]chair_llava15_T500_vcd"
    else:
        # treat as dir or prefix path
        if out_arg.exists() and out_arg.is_dir():
            out_dir = out_arg
            user_prefix = ""  # no user prefix
        else:
            out_dir = out_arg.parent if str(out_arg) != "." else Path(".")
            user_prefix = out_arg.name if out_arg.name else ""

    out_dir.mkdir(parents=True, exist_ok=True)

    tag = "VCD" if args.use_vcd else "BASE"
    noise_tag = f"T{args.noise_step}" if args.use_vcd else "T0"
    n_tag = f"N{args.limit}" if args.limit is not None else "Nall"
    seed_tag = f"seed{args.seed}"

    # 模型名做个安全的短名（避免路径/斜杠）
    model_short = os.path.basename(os.path.expanduser(args.model_path).rstrip("/"))
    model_short = re.sub(r"[^A-Za-z0-9._-]+", "_", model_short)

    auto_name = f"[{tag}]chair_{model_short}_{n_tag}_{noise_tag}_{seed_tag}.jsonl"

    # 如果用户给了前缀，就拼在前面；否则直接用自动名
    final_name = f"{user_prefix}_{auto_name}" if user_prefix else auto_name
    # 去掉可能出现的双下划线/多余下划线
    final_name = re.sub(r"__+", "_", final_name).replace("_.jsonl", ".jsonl")

    out_path = out_dir / final_name
    # ===== end output naming =====


    with out_path.open("w", encoding="utf-8") as f:
        for p in tqdm(img_paths, desc="Generating captions"):
            image_id = coco_image_id_from_name(p.name)

            # build prompt (LLaVA chat)
            qs = args.prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).cuda()

            img = Image.open(p).convert("RGB")
            image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]

            if args.use_vcd:
                image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            else:
                image_tensor_cd = None

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            gen_kwargs = dict(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            if args.top_k is not None:
                gen_kwargs["top_k"] = args.top_k

            if args.use_vcd and image_tensor_cd is not None:
                gen_kwargs.update(
                    images_cd=image_tensor_cd.unsqueeze(0).half().cuda(),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                )

            with torch.inference_mode():
                output_ids = model.generate(**gen_kwargs)

            input_len = input_ids.shape[1]
            caption = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0].strip()
            if caption.endswith(stop_str):
                caption = caption[: -len(stop_str)].strip()

            f.write(json.dumps({"image_id": image_id, "caption": caption}, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[OK] saved -> {out_path}")


if __name__ == "__main__":
    main()
