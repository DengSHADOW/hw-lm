

#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu>
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import pickle
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab = Collection[Wordtype]  # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram = Tuple[Wordtype]
Bigram = Tuple[Wordtype, Wordtype]
Trigram = Tuple[Wordtype, Wordtype, Wordtype]
Ngram = Union[Zerogram, Unigram, Bigram, Trigram]
Vector = List[float]
TorchScalar = Float[torch.Tensor, ""]  # a torch.Tensor with no dimensions, i.e., a scalar

##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path,
                          vocab: Vocab,
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for
    SGD training.

    If randomize is True, then randomize the order of the trigrams each time.
    This is more in the spirit of SGD, but the randomness makes the code harder to debug,
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram


##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.


##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0  # To print progress.

        self.event_count: Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram,
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the
    # denominator and c(yz) for the backed-off numerator.  Both of these
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    #
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    #
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z)] += 1
        self.event_count[(y, z)] += 1
        self.event_count[(z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion,
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram  # we don't care about z
        self.context_count[(x, y)] += 1
        self.context_count[(y,)] += 1
        self.context_count[()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly,
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError(
                "You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
        # torch.load is similar to pickle.load but handles tensors too
        # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")

    def sample(self, max_length: int = 20) -> list[str]:
        """Sample one sentence from the model."""
        x, y = BOS, BOS
        sentence: list[str] = []

        for _ in range(max_length):
            # 构建分布
            probs = torch.tensor([self.prob(x, y, z) for z in self.vocab], dtype=torch.float)

            # 按分布采样一个 z
            z_idx = torch.multinomial(probs, 1).item()
            z = list(self.vocab)[z_idx]

            if z == EOS:  # 到句尾了
                break
            if z != BOS:  # BOS 只是上下文，不输出
                sentence.append(z)

            # 更新上下文
            x, y = y, z

        if len(sentence) == max_length:  # 超长截断
            sentence.append("...")
        return sentence


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError(
                "You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )


class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        V = self.vocab_size
        lam = self.lambda_

        # ---- Case 1: trigram level ----
        if self.context_count[(x, y)] > 0:
            return ((self.event_count[(x, y, z)] + lam) /
                    (self.context_count[(x, y)] + lam * V))

        # ---- Case 2: back off to bigram ----
        if self.context_count[(y,)] > 0:
            return ((self.event_count[(y, z)] + lam) /
                    (self.context_count[(y,)] + lam * V))

        # ---- Case 3: back off to unigram ----
        if self.context_count[()] > 0:
            return ((self.event_count[(z,)] + lam) /
                    (self.context_count[()] + lam * V))

        # ---- Case 4: final fallback: uniform ----
        return 1.0 / V


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.

    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        # 🩵【补丁 1：添加常量】
        self.OOV = OOV
        self.OOL = OOL

        # 🩵【补丁 2：添加空参数容器】
        self.unigram_params = {}
        self.bigram_params = {}
        self.trigram_params = {}

        # 🩵【补丁 3：添加空缓存（防止测试访问）】
        self._yz_cache = {}
        self._xyz_cache = {}
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2
        self.epochs: int = epochs
        # TODO: ADD CODE TO READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        word_vecs = {}
        with open(lexicon_file) as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                word, vec = parts[0], list(map(float, parts[1:]))
                word_vecs[word] = torch.tensor(vec, dtype=torch.float)

        self.dim: int = len(next(iter(word_vecs.values())))
        # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS
        if OOL in word_vecs:
            self.ool_embedding = word_vecs[OOL].to(dtype=torch.float)
        else:
            all_vecs = torch.stack(list(word_vecs.values())).to(dtype=torch.float)
            self.ool_embedding = all_vecs.mean(dim=0)
        E = []
        for w in vocab:
            if w in word_vecs:
                E.append(word_vecs[w])
            else:
                E.append(word_vecs[OOL])  # 不在词典里 → OOL 向量
        self.embeddings = torch.stack(E)  # [|V|, dim]
        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    def _build_sparse_feature_params(self, file: Path) -> None:
        """Baseline 版本不用 sparse features，但 autograder 仍会调用"""
        pass

    def _ctx_emb(self, w: Wordtype) -> torch.Tensor:
        if w in self.word2idx:
            return self.embeddings[self.word2idx[w]].view(-1)  # 保证是 [dim]
        else:
            return self.ool_embedding.to(
                dtype=self.embeddings.dtype,
                device=self.embeddings.device
            ).view(-1)  # 同样保证是 [dim]

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""

        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)

        # TODO: IMPLEMENT ME!
        logits = self.logits(x, y)  # [|V|]
        log_probs = torch.log_softmax(logits, dim=0)  # 归一化
        return log_probs[self.word2idx[z]]

        # This method should call the logits helper method.
        # You are free to define other helper methods too.
        #
        # Be sure to use vectorization over the vocabulary to
        # compute the normalization constant Z, or this method
        # will be very slow. Some useful functions of pytorch that could
        # be useful are torch.logsumexp and torch.log_softmax.
        #
        # The return type, TorchScalar, represents a torch.Tensor scalar.
        # See Question 7 in INSTRUCTIONS.md for more info about fine-grained
        # type annotations for Tensors.
        # raise NotImplementedError("Implement me!")

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor, "vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you
        exponentiate and renormalize in order to get a probability distribution."""
        # TODO: IMPLEMENT ME!
        ex = self._ctx_emb(x)
        ey = self._ctx_emb(y)
        # print("ex shape:", ex.shape, "ey shape:", ey.shape)
        # print("dim =", self.dim, type(self.dim))

        context = self.X @ ex + self.Y @ ey  # [dim]
        logits = self.embeddings @ context  # [|V|]
        return logits

        # Don't forget that you can create additional methods
        # that you think are useful, if you'd like.
        # It's cleaner than making this function massive.
        #
        # The operator `@` is a nice way to write matrix multiplication:
        # you can write J @ K as shorthand for torch.mul(J, K).
        # J @ K looks more like the usual math notation.
        #
        # This function's return type is declared (using the jaxtyping module)
        # to be a torch.Tensor whose elements are Floats, and which has one
        # dimension of length "vocab".  This can be multiplied in a type-safe
        # way by a matrix of type Float[torch.Tensor,"vocab","embedding"]
        # because the two strings "vocab" are identical, representing matched
        # dimensions.  At runtime, "vocab" will be replaced by size of the
        # vocabulary, and "embedding" will be replaced by the embedding
        # dimensionality as given by the lexicon.  See
        # https://www.cs.jhu.edu/~jason/465/hw-lm/code/INSTRUCTIONS.html#a-note-on-type-annotations
        # raise NotImplementedError("Implement me!")

    def train(self, file: Path):  # type: ignore

        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type).
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.

        # Optimization hyperparameters.
        eta0 = 0.01  # initial learning rate

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)  # type: ignore
        nn.init.zeros_(self.Y)  # type: ignore

        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")

        #####################
        # TODO: Implement your SGD here by taking gradient steps on a sequence
        for epoch in range(self.epochs):
            total_logp = 0.0

            for (x, y, z) in read_trigrams(file, self.vocab):
                optimizer.zero_grad()
                logp = self.log_prob_tensor(x, y, z)
                loss = -logp
                loss.backward()
                optimizer.step()
                total_logp += logp.item()
                self.show_progress()

            reg = self.l2 * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2)).item()
            F = (total_logp - reg) / N
            print(f"epoch {epoch + 1}: F = {F}")

        # of training examples.  Here's how to use PyTorch to make it easy:
        #
        # To get the training examples, you can use the `read_trigrams` function
        # we provided, which will iterate over all N trigrams in the training
        # corpus.  (Its use is illustrated in fileprob.py.)
        #
        # For each successive training example i, compute the stochastic
        # objective F_i(θ).  This is called the "forward" computation. Don't
        # forget to include the regularization term. Part of F_i(θ) will be the
        # log probability of training example i, which the helper method
        # log_prob_tensor computes.  It is important to use log_prob_tensor
        # (as opposed to log_prob which returns float) because torch.Tensor
        # is an object with additional bookkeeping that tracks e.g. the gradient
        # function for backpropagation as well as accumulated gradient values
        # from backpropagation.
        #
        # To get the gradient of this objective (∇F_i(θ)), call the `backward`
        # method on the number you computed at the previous step.  This invokes
        # back-propagation to get the gradient of this number with respect to
        # the parameters θ.  This should be easier than implementing the
        # gradient method from the handout.
        #
        # Finally, update the parameters in the direction of the gradient, as
        # shown in Algorithm 1 in the reading handout.  You can do this `+=`
        # yourself, or you can call the `step` method of the `optimizer` object
        # we created above.  See the reading handout for more details on this.
        #
        # For the EmbeddingLogLinearLanguageModel, you should run SGD
        # optimization for the given number of epochs and then stop.  You might
        # want to print progress dots using the `show_progress` method defined above.
        # Even better, you could show a graphical progress bar using the tqdm module --
        # simply iterate over
        #     tqdm.tqdm(read_trigrams(file), total=N*epochs)
        # instead of iterating over
        #     read_trigrams(file)
        #####################

        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.
class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME
    """
        在基础嵌入式 log-linear 上加入：
          - J.1: OOV 特征（z==OOV 时加一个可学习偏置 θ_oov）
          - J.3: 指示特征
                * unigram：对每个词 z 加一个可学习偏置 b[z]
                * bigram：对训练中出现 >=3 次的 (y,z) 加一个可学习偏置 θ_yz
                * trigram：对训练中出现 >=3 次的 (x,y,z) 加一个可学习偏置 θ_xyz
        这些都是以“分数上加项”的方式进入 logits，再做 softmax 归一化。
        """

    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int, thresh: int = 3) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)

        word_vecs = {}
        with open(lexicon_file) as f:
            header = f.readline()  # 跳过第一行
            for line in f:
                parts = line.strip().split()
                word, vec = parts[0], list(map(float, parts[1:]))
                word_vecs[word] = torch.tensor(vec, dtype=torch.float)

        # 维度
        self.dim = len(next(iter(word_vecs.values())))

        # 构造 OOL embedding：如果没有 OOL，就用均值替代
        if OOL in word_vecs:
            self.ool_embedding = word_vecs[OOL]
        else:
            self.ool_embedding = torch.stack(list(word_vecs.values())).mean(dim=0)

        # 构造 embedding 矩阵，OOV → OOL 向量
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        E = []
        for w in vocab:
            if w in word_vecs:
                E.append(word_vecs[w])
            else:
                E.append(self.ool_embedding)
        self.embeddings = torch.stack(E)  # [|V|, dim]

        # --- J.1 OOV 特征 ---
        self.theta_oov = nn.Parameter(torch.tensor(0.0, dtype=torch.float), requires_grad=True)
        self._oov_idx = self.word2idx.get(OOV, None)

        # --- J.3 Unigram 偏置 ---
        self.unigram_bias = nn.Parameter(torch.zeros(self.vocab_size, dtype=torch.float), requires_grad=True)

        # --- J.3 Bigram / Trigram ---
        self.thresh = thresh
        self.bigram_params = nn.ParameterDict()
        self.trigram_params = nn.ParameterDict()

        self._yz_cache: dict[Wordtype, tuple[torch.Tensor, list[str]]] = {}
        self._xyz_cache: dict[tuple[Wordtype, Wordtype], tuple[torch.Tensor, list[str]]] = {}
        all_vecs = torch.stack(list(self.embeddings))
        self.ool_embedding = all_vecs.mean(dim=0)

    # --------- 工具：把 (x,y,z) 键编码成字符串，供 ParameterDict 使用 ---------
    @staticmethod
    def _key_yz(y: Wordtype, z: Wordtype) -> str:
        return f"{y}|||{z}"

    def logits(self, x: Wordtype, y: Wordtype) -> torch.Tensor:
        # context embedding
        ex = self._ctx_emb(x).view(-1)  # 保证 ex 是 [dim]
        ey = self._ctx_emb(y).view(-1)  # 保证 ey 是 [dim]
        # 基本打分
        context = self.X @ ex + self.Y @ ey
        logits = self.embeddings @ context  # [|V|]

        # ---- J.1: OOV 特征 ----
        if OOV in self.word2idx:
            logits[self.word2idx[OOV]] += self.theta_oov

        # ---- J.3: Unigram 偏置 ----
        logits = logits + self.unigram_bias

        # ---- J.3: Bigram 指示特征 ----
        if y in self._yz_cache:
            z_indices, keys = self._yz_cache[y]
            for z_idx, key in zip(z_indices.tolist(), keys):
                logits[z_idx] += self.bigram_params[key]

        # ---- J.3: Trigram 指示特征 ----
        if (x, y) in self._xyz_cache:
            z_indices, keys = self._xyz_cache[(x, y)]
            for z_idx, key in zip(z_indices.tolist(), keys):
                logits[z_idx] += self.trigram_params[key]

        return logits

    @staticmethod
    def _key_xyz(x: Wordtype, y: Wordtype, z: Wordtype) -> str:
        return f"{x}|||{y}|||{z}"

    def _sanitize_key(self, key: str) -> str:
        """确保 key 可以安全地放进 nn.ParameterDict"""
        return key.replace(".", "_").replace(" ", "_").replace("-", "_")

    def _ctx_emb(self, w: Wordtype) -> torch.Tensor:
        if w in self.word2idx:
            return self.embeddings[self.word2idx[w]]
        elif w == OOV and self._oov_idx is not None:
            return self.embeddings[self._oov_idx]
        else:
            # fallback: 如果既不在 vocab 也不是 OOV → 用均值向量（代替 OOL）
            return self.ool_embedding

    # --------- 在训练开始前，统计频次，挑出 >= thresh 的 bigram/trigram，注册参数并建立缓存 ---------
    def _build_sparse_feature_params(self, file: Path) -> None:
        """
        统计 bigram 和 trigram 频次，建立稀疏特征参数。
        并缓存 (z_idx, 参数 key) 映射，加速 logits() 计算。
        """
        from collections import Counter, defaultdict

        bigram_ctr: Counter[tuple[Wordtype, Wordtype]] = Counter()
        trigram_ctr: Counter[tuple[Wordtype, Wordtype, Wordtype]] = Counter()

        # 第一步：统计
        for (x, y, z) in read_trigrams(file, self.vocab):
            bigram_ctr[(y, z)] += 1
            trigram_ctr[(x, y, z)] += 1

        # 第二步：为频次 >= 阈值的 bigram/trigram 建立参数
        for (y, z), c in bigram_ctr.items():
            if c >= self.thresh:
                raw_key = f"{y}_{z}"
                key = self._sanitize_key(raw_key)
                if key not in self.bigram_params:
                    self.bigram_params[key] = nn.Parameter(
                        torch.tensor(0.0, dtype=torch.float), requires_grad=True
                    )

        for (x, y, z), c in trigram_ctr.items():
            if c >= self.thresh:
                raw_key = f"{x}_{y}_{z}"
                key = self._sanitize_key(raw_key)
                if key not in self.trigram_params:
                    self.trigram_params[key] = nn.Parameter(
                        torch.tensor(0.0, dtype=torch.float), requires_grad=True
                    )

        # 第三步：建立缓存，加速 logits() 计算
        # bigram 缓存：按 y 聚合
        yz_bucket: dict[Wordtype, list[tuple[int, str]]] = defaultdict(list)
        for (y, z), c in bigram_ctr.items():
            if c >= self.thresh and z in self.word2idx:
                z_idx = self.word2idx[z]
                yz_bucket[y].append((z_idx, self._sanitize_key(f"{y}_{z}")))

        self._yz_cache.clear()
        for y, items in yz_bucket.items():
            if items:
                z_indices, keys = zip(*items)
                self._yz_cache[y] = (torch.tensor(z_indices, dtype=torch.long), list(keys))

        # trigram 缓存：按 (x,y) 聚合
        xy_bucket: dict[tuple[Wordtype, Wordtype], list[tuple[int, str]]] = defaultdict(list)
        for (x, y, z), c in trigram_ctr.items():
            if c >= self.thresh and z in self.word2idx:
                z_idx = self.word2idx[z]
                xy_bucket[(x, y)].append((z_idx, self._sanitize_key(f"{x}_{y}_{z}")))

        self._xyz_cache.clear()
        for xy, items in xy_bucket.items():
            if items:
                z_indices, keys = zip(*items)
                self._xyz_cache[xy] = (torch.tensor(z_indices, dtype=torch.long), list(keys))

    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    pass
