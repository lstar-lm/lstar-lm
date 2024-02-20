from functools import cache

import lstar

from lstar_lm.dfa_sat_learner import guess_dfa_sat, distinguishing_query


class OutOfQueries(Exception):
    """Thrown if learner acts too many membership queries."""


def guess_dfa_lstar(positive, negative, alphabet, oracle, n_queries):
    # Bound the number of membership queries L* can perform and call
    # dfa_identify on remaining.
    positive, negative = set(positive), set(negative)

    @cache
    def wrapped_oracle(word):
        nonlocal n_queries
        nonlocal positive
        nonlocal negative

        if n_queries == 0: raise OutOfQueries
        n_queries -= 1

        label = oracle(word) 
        if label is True:    positive.add(word)
        elif label is False: negative.add(word)
        return label


    def find_counter_example(lang):
        nonlocal n_queries
        for word in positive:
            if lang.label(word) is False:
                return word
        for word in negative:
            if lang.label(word) is True:
                return word

        for _ in range(n_queries):
            word = distinguishing_query(positive, negative, alphabet)
            if word is None:
                continue  # wasted a query on something we can't label.
            if lang.label(word) != wrapped_oracle(word):
                return word
        return None

    try:
        return lstar.learn_dfa(
            inputs=alphabet,
            # L* doesn't support unsure labels. Thus map unsure -> False.
            label=lambda w: wrapped_oracle(w) is False,  
            find_counter_example=find_counter_example
        ).normalize()
    except:
        return guess_dfa_sat(positive, negative, alphabet, wrapped_oracle, 0)

