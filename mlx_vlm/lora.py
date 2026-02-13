import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Prefer importing the *local* mlx_vlm package (next to this script) rather than
# an installed site-packages version.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Work around occasional brotli decoder crashes seen in some aiohttp/httpx stacks
# (e.g. during dataset/image fetching in multiprocessing). Disabling aiohttp
# optional C-extensions prevents brotli usage.
os.environ.setdefault("AIOHTTP_NO_EXTENSIONS", "1")

import mlx.optimizers as optim
from datasets import load_dataset
from tqdm import tqdm

# NOTE: this file is being run as a standalone script (e.g. `python lora.py`).
# Relative imports (from .something import ...) only work when executed as a package
# module (e.g. `python -m mlx_vlm.lora`).
#
# To make this script runnable from arbitrary locations, we import from the
# installed `mlx_vlm` package instead.
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.trainer import Dataset, Trainer, save_adapter
from mlx_vlm.trainer.utils import apply_lora_layers, find_all_linear_names, get_peft_model
from mlx_vlm.utils import load, load_image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def main(args):
    logger.info(f"\033[32mLoading model from {args.model_path}\033[0m")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    image_processor = load_image_processor(args.model_path)

    logger.info(f"\033[32mLoading dataset from {args.dataset}\033[0m")
    # Hugging Face Datasets will materialize Arrow caches; large datasets (e.g. ImageNet)
    # can easily exceed free space on the system drive. Allow overriding the cache dir.
    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)

    # Expected columns for mlx-vlm LoRA training are typically:
    # - messages (or conversations): chat-style supervision
    # - images (or image): image(s) per example
    #
    # Some datasets (e.g. ImageNet) have `label` + `image` instead. In that case we
    # synthesize a minimal chat conversation from the label.
    if "messages" not in dataset.column_names and "conversations" not in dataset.column_names:
        if "label" in dataset.column_names:
            logger.warning(
                "Dataset has no 'messages' column; found 'label'. Synthesizing messages from label."
            )

            def label_to_messages(example):
                # Try to turn ClassLabel -> string name if available
                label_val = example.get("label")
                try:
                    feat = dataset.features.get("label")
                    if hasattr(feat, "int2str"):
                        label_text = feat.int2str(label_val)
                    else:
                        label_text = str(label_val)
                except Exception:
                    label_text = str(label_val)

                # Minimal chat format: user asks about the image; assistant answers label.
                example["messages"] = [
                    {"role": "user", "content": "Describe the image."},
                    {"role": "assistant", "content": label_text},
                ]
                return example

            dataset = dataset.map(label_to_messages)
        else:
            raise ValueError("Dataset must have a 'messages'/'conversations' column (or a 'label' column to synthesize messages)")

    if "images" not in dataset.column_names and "image" not in dataset.column_names:
        raise ValueError("Dataset must have an 'images' or 'image' column")

    if args.apply_chat_template:
        logger.info(f"\033[32mApplying chat template to the dataset\033[0m")

        def process_data(examples):
            if config["model_type"] == "pixtral":
                conversations = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
                examples["messages"] = [
                    json.dumps(item, ensure_ascii=False) for item in conversations
                ]
            else:
                examples["messages"] = apply_chat_template(
                    config=config,
                    processor=processor,
                    prompt=examples["messages"],
                    return_messages=True,
                )
            return examples

        # HF Datasets map() is single-process by default. Use num_proc>1 to
        # parallelize CPU-bound preprocessing (JSON/template formatting).
        num_proc = args.num_proc
        if num_proc == 0:
            num_proc = os.cpu_count() or 1
        try:
            dataset = dataset.map(
                process_data,
                num_proc=num_proc if num_proc > 1 else None,
            )
        except Exception as e:
            logger.warning(
                "dataset.map() failed (num_proc=%s): %s. Retrying single-process.",
                num_proc,
                e,
            )
            dataset = dataset.map(process_data, num_proc=None)

    dataset = Dataset(
        dataset,
        config,
        processor,
        image_processor=image_processor,
        image_resize_shape=args.image_resize_shape,
    )

    adapter_path = args.adapter_path
    if adapter_path:
        logger.info(f"\033[32mResuming from adapter path {adapter_path}\033[0m")
        logger.info(
            f"\033[32mLora rank, alpha, and dropout will be loaded from adapter_config.json file\033[0m"
        )

        model = apply_lora_layers(model, adapter_path)

    else:
        logger.info(f"\033[32mSetting up LoRA\033[0m")

        list_of_modules = find_all_linear_names(model.language_model)
        model = get_peft_model(
            model,
            list_of_modules,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

    logger.info(f"\033[32mSetting up optimizer\033[0m")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info(f"\033[32mSetting up trainer\033[0m")
    trainer = Trainer(model, optimizer)

    model.train()

    # Training loop
    logger.info(f"\033[32mTraining model\033[0m")
    for epoch in range(args.epochs):
        if args.steps == 0:
            args.steps = len(dataset) // args.batch_size

        progress_bar = tqdm(range(args.steps), position=0, leave=True)
        for i in progress_bar:
            loss = trainer.train_step(
                dataset[i * args.batch_size : (i + 1) * args.batch_size]
            )
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"Epoch": epoch, "Step": i, "Loss": f"{loss.item():.4f}"}
            )

            if i % args.print_every == 0:
                # Log additional information
                custom_print(
                    {
                        "Epoch": epoch,
                        "Step": i,
                        "Loss": f"{loss.item():.4f}",
                    }
                )
        # Save the interim adapter after each epoch except the last.
        if args.save_after_epoch and (epoch < (args.epochs - 1)):
            head, tail = os.path.split(args.output_path)
            save_adapter(model, head + os.sep + "epoch_" + str(epoch) + "_" + tail)

    # Save the adapter
    save_adapter(model, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NanoLLaVA model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Qwen2-VL-2B-Instruct-bf16",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Cache directory for Hugging Face Datasets (Arrow cache, downloads). "
            "Use a drive with lots of free space (e.g. /Volumes/QuadX/...)."
        ),
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Split to use for training"
    )
    parser.add_argument(
        "--image-resize-shape",
        type=int,
        nargs=2,
        default=None,
        help="Resize images to this shape",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_false",
        help="Apply chat template to the dataset",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train"
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--print-every", type=int, default=10, help="Print loss every n steps"
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=0.1,
        help="LoRA scaling factor (alpha / rank)",
    )
    parser.add_argument("--lora-rank", type=int, default=10, help="LoRA rank")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--output-path",
        type=str,
        default="adapters",
        help="Path to save the trained adapter",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Load path to resume training from a previously saved adapter",
    )
    parser.add_argument(
        "--save-after-epoch",
        action="store_true",
        help="Save interim versions of adapter files after each epoch",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=0,
        help=(
            "Number of processes for HF Datasets preprocessing (dataset.map). "
            "0 = auto (os.cpu_count()), 1 = disable multiprocessing."
        ),
    )

    args = parser.parse_args()
    main(args)
