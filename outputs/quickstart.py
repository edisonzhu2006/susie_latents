"""SUSIE quickstart: generate a subgoal image from (current image, prompt).

Outputs (next to this script, i.e. susie_latents/outputs/):
  input.png       - the original 256x256 image
  subgoal.png     - model's predicted subgoal
  sidebyside.png  - input | subgoal concatenated for easy comparison

Usage:
  python quickstart.py
  python quickstart.py --prompt "pick up the spoon"
  python quickstart.py --image_url <url> --prompt "..."
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from susie.model import create_sample_fn  # noqa: E402
from susie.jax_utils import initialize_compilation_cache  # noqa: E402

DEFAULT_IMAGE_URL = (
    "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/"
    "datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/"
    "images0/im_12.jpg"
)
DEFAULT_PROMPT = "open the drawer"
OUT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default="kvablack/susie")
    parser.add_argument(
        "--pretrained",
        default="lodestones/stable-diffusion-v1-5-flax",
        help="HF path for VAE+text_encoder (flax weights). "
        "runwayml/stable-diffusion-v1-5:flax no longer exists.",
    )
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--prompt_w", type=float, default=7.5)
    parser.add_argument("--context_w", type=float, default=2.5)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"[1/4] Loading model {args.model} ...")
    initialize_compilation_cache()
    sample_fn = create_sample_fn(
        args.model,
        num_timesteps=args.num_timesteps,
        prompt_w=args.prompt_w,
        context_w=args.context_w,
        pretrained_path=args.pretrained,
    )

    print(f"[2/4] Fetching image: {args.image_url}")
    raw = requests.get(args.image_url, stream=True, timeout=30).raw
    image = np.array(Image.open(raw).resize((256, 256)))
    assert image.shape == (256, 256, 3), f"unexpected shape: {image.shape}"

    print(f"[3/4] Sampling subgoal for prompt: {args.prompt!r}")
    image_out = sample_fn(image, args.prompt)

    print("[4/4] Saving outputs ...")
    Image.fromarray(image).save(OUT_DIR / "input.png")
    Image.fromarray(image_out).save(OUT_DIR / "subgoal.png")
    combined = np.concatenate([image, image_out], axis=1)
    Image.fromarray(combined).save(OUT_DIR / "sidebyside.png")

    print(f"Done. Output files in {OUT_DIR}:")
    for p in sorted(OUT_DIR.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
