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

def check_error(file, model) -> bool: # check model = file for error rate (current works for gen/spam, en/sp)
    modelName = model.split('.')[0]
    fileName = file.name.split('.')[0]
    return fileName != modelName

def inv_logit_neg(a: float) -> float:
    # computes 1 / (1 + exp(a)) stably
    if a >= 0:
        t = math.exp(-a)      # tiny or 0.0, never overflows
        return t / (1.0 + t)
    else:
        return 1.0 / (1.0 + math.exp(a))  # safe since exp(a) <= 1


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    # check device availability
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            log.critical("MPS device not available")
            exit(1)
    torch.set_default_device(args.device)

    # Validate prior
    if not (0.0 < args.prior < 1.0):
        log.critical("prior must be in (0,1)")
        raise SystemExit(1)
    
    # load models
    lm1 = LanguageModel.load(args.model1, device=args.device)
    lm2 = LanguageModel.load(args.model2, device=args.device)

    # Vocab must match
    if set(lm1.vocab) != set(lm2.vocab):
        log.critical("The two models must share the same vocabulary. "
                     "Re-train both with the SAME vocab file.")
        raise SystemExit(1)

    prior1 = args.prior
    prior2 = 1.0 - prior1

    log_prior1 = math.log(prior1)
    log_prior2 = math.log(prior2)

    # Q3(c): track Δ(d) = log P(d|model1) - log P(d|model2)
    max_delta = float('-inf')
    min_delta = float('inf')

    count1, count2 = 0, 0
    total_files = len(args.test_files)

    e: int = 0
    for file in args.test_files:
        fp1 = file_log_prob(file, lm1)
        fp2 = file_log_prob(file, lm2)

        log_prob1 = fp1 + log_prior1
        log_prob2 = fp2 + log_prior2

        if log_prob1 >= log_prob2:
            model = args.model1.name
            print(f"{args.model1.name} {file.name}")
            count1 += 1
        else:
            model = args.model2.name
            print(f"{args.model2.name} {file.name}")
            count2 += 1
        
        # count error rate
        if(check_error(file, model)):
            e = e + 1
            
        # compute Δ using likelihoods only (no priors)
        delta = fp1 - fp2               # Δ(d)

        if delta > max_delta:
            max_delta = delta
        if delta < min_delta:
            min_delta = delta
        
    if total_files > 0:
        pct1 = 100.0 * count1 / total_files
        pct2 = 100.0 * count2 / total_files
        print(f"{count1} files were more probably from {args.model1.name} ({pct1:.2f}%)")
        print(f"{count2} files were more probably from {args.model2.name} ({pct2:.2f}%)")

    # Print out error rate
    #print("Error rate: ", e/total_files)
    
    # thresholds derived from Δ
    # Smallest p(gen) that forces ALL files → spam:
    p_star   = inv_logit_neg(max_delta)
    #print(f"Q3(c): max  = {max_delta:.4f}   p for ALL to spam = {p_star:.6f}")
    

if __name__ == "__main__":
    main()
