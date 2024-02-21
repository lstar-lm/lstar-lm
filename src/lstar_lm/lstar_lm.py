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
              task_description="",
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
    """L*LM implementation without demonstration learning modality."""
    # 1. Initialize LLM oracle.
    label = L.llm_oracle(positive, negative, 
                       desc=task_description, verbose=verbose,
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


def dfa_search_with_diss(*,
                         alphabet,

                         # Passive identification params.
                         positive=(),
                         negative=(),

                         # LLM oracle params.
                         task_description="",
                         llm_params=L.DEFAULT_PARAMS,
                         llm_query_callback=lambda *_: None,
                         llm_endpoint=L.DEFAULT_ENDPOINT,
                         allow_unsure=True,

                         # Active learning params.
                         ce_search_depth=-1,
                         random_iters=0,    # Only used for random search.
                         use_random_search=True,
                         active_queries=10,
                         use_dfa_identify=True,

                         # Demonstration learning parameters.
                         to_chain,
                         demonstrations,
                         max_diss_iters: int,
                         diss_params,  # Other DISS parameters.

                         # OTHER
                         verbose=False,
              ):
    """L*LM implementation with demonstration learning modality."""
    try:
        import diss
        from diss.concept_classes.dfa_concept import DFAConcept
    except ModuleNotFoundError as e:
        print("Please install the DISS package.\npip install diss")
        raise e

    def llm_identifer(data, concept=None):
        lang = guess_dfa(positive=data.positive,
                         negative=data.negative, 
                         alphabet=alphabet,
                         task_description=task_description,
                         ce_search_depth=ce_search_depth,
                         verbose=verbose,
                         allow_unsure=allow_unsure,
                         llm_query_call_back=llm_query_callback,
                         llm_endpoint=llm_endpoint,
                         use_random_search=use_random_search,
                         random_iters=random_iters,
                         active_queries=active_queries,
                         use_dfa_identify=use_dfa_identify)

        return DFAConcept.from_dfa(lang)

    diss_params = {
        "demos": demonstrations,
        "sgs_temp": 0.01,
        "n_iters": max_diss_iters,
        "surprise_weight": 1,
        "reset_period": 30,
        "size_weight": 1/80,
        "example_drop_prob": 1/20,
        "synth_timeout": 0,
        "competency": lambda *_: 10,
        "to_chain": to_chain,
        "to_concept": llm_identifer,
    } | diss_params

    return diss.diss(**diss_params)

__all__ = ['dfa_search_with_diss', 'guess_dfa']
