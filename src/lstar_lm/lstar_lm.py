import random
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

import funcy as fn

import lstar_lm as L


def random_search(alphabet, max_depth, samples):
    words = L.all_words(alphabet)
    words = fn.takewhile(lambda w: len(w) < max_depth, words)
    words = list(words)
    if len(words) < samples: return words
    yield from random.sample(words, samples)


def guess_dfa(positive,
              negative,
              alphabet,
              desc="",
              llm_params=L.DEFAULT_PARAMS,
              ce_search_depth=-1,
              random_iters=0,    # Only used for random search.
              active_queries=10,
              use_random_search=True,
              allow_unsure=True,
              verbose=False,
              llm_query_call_back=lambda *_: None,
              use_dfa_identify=True,
              llm_endpoint=L.DEFAULT_ENDPOINT):
    # 1. Initialize LLM oracle.
    label = L.llm_oracle(positive, negative, 
                       desc=desc, verbose=verbose,
                       params=llm_params,
                       allow_unsure=allow_unsure,
                       llm_query_call_back=llm_query_call_back,
                       endpoint=llm_endpoint)

    # 2. Augment labeled examples with labeled examples.
    words = random_search(alphabet=alphabet,
                          max_depth=ce_search_depth,
                          samples=random_iters)

    positive, negative = set(positive), set(negative)
    for word in words:
        bucket = positive if label(word) else negative
        bucket.add(word)

    # 3. Run learner.
    learner = L.guess_dfa_sat if use_dfa_identify else L.guess_dfa_lstar
    return learner(positive, negative, set(alphabet), label, active_queries)
