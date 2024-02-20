from functools import cache, partial
from itertools import combinations_with_replacement

import funcy as fn
from dfa_identify import find_dfas


find_dfas = partial(find_dfas, order_by_stutter=True, allow_unminimized=True)


def all_words(alphabet):
    """Enumerates all words in the alphabet."""
    n = 0
    while True:
        yield from combinations_with_replacement(alphabet, n)
        n += 1


def distinguishing_query(positive, negative, alphabet):
    """Return a string that seperates the two smallest (consistent) DFAs."""
    candidates = find_dfas(positive, negative, alphabet=alphabet)
    lang1 = next(candidates)

    # DFAs may represent the same language. Filter those out.
    candidates = (c for c in candidates if lang1 != c)
    lang2 = next(candidates, None)

    # Try to find a seperating word.
    if (lang1 is not None) and (lang2 is not None):
        return tuple((lang1 ^ lang2).find_word(True))

    # TODO: due to  a bug in dfa-identify allow_unminimized doesn't always work
    # so we need to come up with a word that is not in positive/negative but is
    # not constrained by positive / negative.
    constrained = set(positive) | set(negative)
    return fn.first(w for w in all_words(alphabet) if w not in constrained)


def guess_dfa_sat(positive, negative, alphabet, oracle, n_queries): 
    """
    SAT based version space learner that actively queries the oracle
    for a string that distinguishes two "minimal" DFAs, where
    minimal is lexicographically ordered in (#states, #edges).
    """
    positive, negative = list(positive), list(negative)

    # 1. Ask membership queries that distiguish remaining candidates.
    for _ in range(n_queries):
        word = distinguishing_query(positive, negative, alphabet)

        label = oracle(word)
        if label is True:    positive.append(word)
        elif label is False: negative.append(word)
        else: assert label is None  # idk case.

    # 2. Return minimal consistent DFA.
    # TODO: consider sampling based on size.
    return next(find_dfas(positive, negative, alphabet=alphabet))


__all__ = [ 'guess_dfa_sat', 'all_words', 'distinguishing_query' ]
