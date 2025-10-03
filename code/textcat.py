#!/usr/bin/env python3
"""
Text categorization using two language models and Bayes' theorem.
"""

import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, read_trigrams, num_tokens

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model1", type=Path, help="path to first trained model (e.g., gen.model)")
    parser.add_argument("model2", type=Path, help="path to second trained model (e.g., spam.model)")
    parser.add_argument("prior", type=float, help="prior probability of first model (between 0 and 1)")
    parser.add_argument("test_files", type=Path, nargs="*", help="files to classify")

    parser.add_argument("--device", type=str, default="cpu",
                        choices=['cpu','cuda','mps'],
                        help="device to use for PyTorch")

    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG)
    verbosity.add_argument("-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING)

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """Return log P(file | model)"""
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)
        if log_prob == -math.inf:
            break
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # check device availability
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            log.critical("MPS device not available")
            exit(1)
    torch.set_default_device(args.device)

    # load models
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)

    prior1 = args.prior
    prior2 = 1.0 - prior1

    log_prior1 = math.log(prior1)
    log_prior2 = math.log(prior2)

    count1, count2 = 0, 0
    total_files = len(args.test_files)

    for file in args.test_files:
        log_prob1 = file_log_prob(file, lm1) + log_prior1
        log_prob2 = file_log_prob(file, lm2) + log_prior2

        if log_prob1 >= log_prob2:
            print(f"{args.model1.name} {file.name}")
            count1 += 1
        else:
            print(f"{args.model2.name} {file.name}")
            count2 += 1

    if total_files > 0:
        pct1 = 100.0 * count1 / total_files
        pct2 = 100.0 * count2 / total_files
        print(f"{count1} files were more probably from {args.model1.name} ({pct1:.2f}%)")
        print(f"{count2} files were more probably from {args.model2.name} ({pct2:.2f}%)")


if __name__ == "__main__":
    main()
