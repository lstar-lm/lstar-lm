import random
from itertools import combinations_with_replacement

import funcy as fn

from lstar_lm.llm import llm_oracle, DEFAULT_PARAMS, DEFAULT_ENDPOINT
from lstar_lm.dfa_sat_learner import all_words, guess_dfa_sat
from lstar_lm.dfa_lstar_learner import guess_dfa_lstar


def random_search(alphabet, max_depth, samples):
    words = all_words(alphabet)
    words = fn.takewhile(lambda w: len(w) < max_depth, words)
    words = list(words)
    if len(words) < samples: return words
    yield from random.sample(words, samples)


def guess_dfa(positive,
              negative,
              alphabet,
              desc="",
              llm_params=DEFAULT_PARAMS,
              hypothesize_rule=True,
              ce_search_depth=-1,
              random_iters=0,    # Only used for random search.
              active_queries=10,
              use_random_search=True,
              allow_unsure=True,
              verbose=False,
              llm_query_call_back=lambda *_: None,
              use_dfa_identify=True,
              llm_endpoint=DEFAULT_ENDPOINT):
    # 1. Initialize LLM oracle.
    label = llm_oracle(positive, negative, 
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
    learner = guess_dfa_sat if use_dfa_identify else guess_dfa_lstar
    return learner(positive, negative, set(alphabet), label, active_queries)
