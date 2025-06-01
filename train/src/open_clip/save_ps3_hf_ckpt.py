import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import torch

try:
    from huggingface_hub import (
        create_repo,
        get_hf_file_metadata,
        hf_hub_download,
        hf_hub_url,
        repo_type_and_id_from_hf_id,
        upload_folder,
        list_repo_files,
    )
    from huggingface_hub.utils import EntryNotFoundError
    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

from .factory import create_model_from_pretrained, get_model_config, get_tokenizer
from .tokenizer import HFTokenizer, DEFAULT_CONTEXT_LENGTH

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'config.json'
HF_PROCESSOR_CONFIG_NAME = 'preprocessor_config.json'
HF_TOKENIZER_CONFIG_NAME = 'tokenizer_config.json'


def save_tokenizer_config(
    model_config,
    save_dir: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    hf_config = {
        "tokenizer_name": model_config["text_cfg"]["hf_tokenizer_name"],
        "context_length": model_config["text_cfg"].get('context_length', DEFAULT_CONTEXT_LENGTH),
        **model_config["text_cfg"].get("tokenizer_kwargs", {}),
    }

    with (save_dir / HF_TOKENIZER_CONFIG_NAME).open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_preprocessor_config(
    model,
    save_dir: str,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    cfg = model.visual.preprocess_cfg

    hf_config = {
        "image_size": cfg["size"],
        "mean": cfg["mean"],
        "std": cfg["std"],
        "interpolation": cfg["interpolation"],
        "resize_mode": cfg["resize_mode"],
        # "fill_color": cfg["fill_color"],
    }

    with (save_dir / HF_PROCESSOR_CONFIG_NAME).open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_config(
    model,
    save_dir: str,
    model_config: Optional[dict],
    save_vision_model_only: bool = False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Vision model config
    vision_hf_config = {
        "architectures": ["PS3VisionModel"],
        "model_type": "ps3_vision_model",
        **model_config["vision_cfg"],
    }

    # Text model config
    text_hf_config = model_config["text_cfg"]
    text_hf_config["architectures"] = ["PS3TextModel"]
    text_hf_config["model_type"] = "ps3_text_model"
    text_hf_config["output_dim"] = model_config["embed_dim"]
    text_hf_config["prompt_proj_dim"] = model.visual.width

    # Merge vision and text configs
    if save_vision_model_only:
        hf_config = vision_hf_config
    else:
        hf_config = {
            "architectures": ["PS3Model"],
            "model_type": "ps3",
            "vision_config": vision_hf_config,
            "text_config": text_hf_config,
        }

    with (save_dir / HF_CONFIG_NAME).open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_model(
    model,
    save_dir: str,
    safe_serialization: Union[bool, str] = True,
    save_vision_model_only: bool = False,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    tensors = model.state_dict()

    # process vision model weights
    tensors = {k.replace("visual.", "vision_model."): v for k, v in tensors.items()}

    # process text model weights
    tensors = {k.replace("text.", "text_model."): v for k, v in tensors.items()}
    tensors = {"text_model." + k if k.startswith("prompt_proj.") else k: v for k, v in tensors.items()}

    if save_vision_model_only:
        tensors = {k: v for k, v in tensors.items() if "vision_model." in k}

    if safe_serialization is True or safe_serialization == "both":
        assert _has_safetensors, "`pip install safetensors` to use .safetensors"
        safetensors.torch.save_file(tensors, save_dir / HF_SAFE_WEIGHTS_NAME, metadata={'format': 'pt'})
    if safe_serialization is False or safe_serialization == "both":
        torch.save(tensors, save_dir / HF_WEIGHTS_NAME)


def save_hf_ckpt(
    model_name,
    pretrained: str,
    save_dir: str,
    save_vision_model_only: bool = False,
    **kwargs,
):
    model, processor = create_model_from_pretrained(
        model_name,
        pretrained=pretrained,
        load_weights_only=False,
    )

    model_config = get_model_config(model_name)

    save_model(
        model,
        save_dir=save_dir,
        save_vision_model_only=save_vision_model_only,
    )

    save_config(
        model,
        save_dir=save_dir, 
        model_config=model_config,
        save_vision_model_only=save_vision_model_only,
    )

    save_preprocessor_config(
        model,
        save_dir=save_dir,
    )

    save_tokenizer_config(
        model_config,
        save_dir=save_dir,
    )


def push_hf_ckpt(repo_id, save_dir, token=None, private=False):
    # Create repo if it doesn't exist yet
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)

    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push to Hugging Face Hub")
    parser.add_argument(
        "--model", type=str, help="Name of the model to use.",
    )
    parser.add_argument(
        "--pretrained", type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--save-dir", type=str,
        help="Which directory to save the model to.",
    )
    parser.add_argument(
        "--save-vision-model-only",
        help="Whether to save the vision model weights only.",
        action="store_true",
    )
    parser.add_argument(
        "--push-to-repo", type=str,
        help="Destination HF Hub repo-id ie 'organization/model_id'.",
        default=None
    )
    args = parser.parse_args()

    save_hf_ckpt(
        args.model,
        args.pretrained,
        args.save_dir,
        save_vision_model_only=args.save_vision_model_only,
    )
    print(f'{args.pretrained} saved to {args.save_dir}.')

    if args.push_to_repo is not None:
        push_hf_ckpt(
            repo_id=args.push_to_repo,
            folder_path=args.save_dir,
        )
        print(f'{args.pretrained} pushed to {args.push_to_repo}.')
