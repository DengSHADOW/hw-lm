#!/usr/bin/env python3
"""
Samples random sentences from a trained trigram language model.
"""

import argparse
import logging
from pathlib import Path
import torch

from probs import LanguageModel

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentences",
        type=int,
        help="how many sentences to sample"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum sentence length (tokens)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu, cuda, or mps on Mac)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Setup device
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            logging.critical("MPS not available on this system.")
            exit(1)
    torch.set_default_device(args.device)

    # Load trained model
    lm = LanguageModel.load(args.model, device=args.device)

    # Sample sentences
    for i in range(args.num_sentences):
        sent = lm.sample(max_length=args.max_length)
        # Join words into a string
        if sent and sent[-1] == "...":
            print(" ".join(sent))   # truncated
        else:
            print(" ".join(sent))


if __name__ == "__main__":
    main()
