import os
import httpx
import lstar
import random
from dfa_identify import find_dfas
from functools import cache, partial
from itertools import combinations_with_replacement


DEFAULT_ENDPOINT ="http://127.0.0.1:8080/completion"

BASE_PROMPT = """
{task_description}

Additionally, by examining demonstrations of the task, we conjecture the following labeled examples:

"""

MEMBERSHIP_PROMPT = """
Please briefly answer the following questions using step-by-step reasoning to
show your work. Do not answer any other question. When you arrive at a
conclusion, please state it as FINAL_ANSWER: <yes, no>.
"""

MEMBERSHIP_GRAMMAR = """
  root ::= work answer
  work ::= [^"FINAL_ANSWER"]+
  answer ::= "FINAL_ANSWER: " ("yes" | "no")
"""

MEMBERSHIP_PROMPT_IDK = """
Please briefly answer the following questions using step-by-step reasoning to
show your work. Do not answer any other question. When you arrive at a
conclusion, please state it as FINAL_ANSWER: <yes, no, unsure>.
"""

MEMBERSHIP_GRAMMAR_IDK = """
  root ::= work answer
  work ::= [^"FINAL_ANSWER"]+
  answer ::= "FINAL_ANSWER: " ("yes" | "no" | "unsure")
"""


def word_to_str(word):
    return f"[{', '.join(word)}]"

def word_to_question(word):
    return f"Is {word_to_str(word)} a positive example?"

def examples_to_str(positive, negative):
    output = "POSITIVE EXAMPLES\n"
    for word in positive:
        output += f"  - {word_to_str(word)}\n"

    output += "\nNEGATIVE EXAMPLES\n"
    for word in negative:
        output += f"  - {word_to_str(word)}\n"
    return output


def parse_membership_reponse(msg):
    match msg.split('FINAL_ANSWER: '):
        case [_, 'yes']:    return True
        case [_, 'unsure']: return None
        case _:             return False


DEFAULT_PARAMS = {
    "cache_prompt": True,
    "image_data": [],
    "mirostat": 0,
    "mirostat_eta": 0.1,
    "mirostat_tau": 5,
    "n_predict": -1,
    "n_probs": 0,
    "presence_penalty": 0,
    "repeat_last_n": 241,
    "repeat_penalty": 1.18,
    "slot_id": 3,
    "stop": ["</s>", "AI:", "User:"],
    #"stream": False,
    "temperature": 0.1,
    "tfs_z": 1,
    "top_k": 40,
    "top_p": 0.5,
    "typical_p": 1,
}


def run_llm(positive=(),
            negative=(),
            desc="",
            verbose=True,
            params=DEFAULT_PARAMS,
            allow_unsure=True,
            llm_query_call_back=lambda *_: None,
            endpoint=DEFAULT_ENDPOINT):
    word = yield
    base_prompt = BASE_PROMPT.format(task_description=desc)
    base_prompt = f"{base_prompt}\n{examples_to_str(positive, negative)}"

    # Membership queries.
    membership_prompt = MEMBERSHIP_PROMPT_IDK if allow_unsure else MEMBERSHIP_PROMPT
    membership_grammar = MEMBERSHIP_GRAMMAR_IDK if allow_unsure else MEMBERSHIP_GRAMMAR
    prompt = base_prompt + membership_prompt
    while True:
        query = word_to_question(word)
        prompt += f"{query}\nAI:"
        data = params | {"prompt": prompt, "grammar": membership_grammar}
        response = httpx.post(ENDPOINT, timeout=1000, json=data).json()
        content = response["content"]
        llm_query_call_back(prompt, content)
        prompt += f"{content}\nUser:"

        if verbose: print(query + f"{content}\nUser:")
        word = yield parse_membership_reponse(content)


def enumerative_search(alphabet, max_depth):
    for n in range(max_depth+1):
        yield from combinations_with_replacement(alphabet, n)


def random_search(alphabet, max_depth, samples):
    words = list(enumerative_search(alphabet, max_depth))
    yield from random.sample(words, samples)


class OutOfQueries(Exception):
    """Thrown if learner acts too many membership queries."""


IDENTIFY_PARAMS = {
    "order_by_stutter": True,
    "allow_unminimized": True
}


def distinguishing_query(positive, negative, alphabet, lang1=None):
    candidates = find_dfas(positive, negative, alphabet=alphabet, **IDENTIFY_PARAMS)
    if lang1 is None:
        lang1 = next(candidates)

    # DFAs may represent the same language. Filter those out.
    candidates = (c for c in candidates if lang1 != c)
    lang2 = next(candidates, None)

    # Try to find a seperating word.
    if (lang1 is not None) and (lang2 is not None):
        try:
            return tuple((lang1 ^ lang2).find_word(True))
        except:
            print(lang1)
            print(lang2)
            import pdb; pdb.set_trace()

    # TODO: due to  a bug in dfa-identify allow_unminimized doesn't always work
    # so we need to come up with a word that is not in positive/negative but is
    # labeled differently to our candidate dfa.
    #
    # We do this by searching over random strings.
    constrained = set(positive) | set(negative)
    depth = 10
    while True:
        for word in random_search(alphabet, 10, 10):
            if word in constrained:
                continue
            return word
        depth += 1


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


def guess_dfa_sat(positive, negative, alphabet, oracle, n_queries): 
    positive, negative = list(positive), list(negative)

    # 1. Ask membership queries that distiguish remaining candidates.
    for _ in range(n_queries):
        print(_)
        word = distinguishing_query(positive, negative, alphabet)

        label = oracle(word)
        if label is True:    positive.append(word)
        elif label is False: negative.append(word)
        else: assert label is None  # idk case.

    # 2. Return minimal consistent DFA.
    # TODO: consider sampling based on size.
    return next(find_dfas(positive, negative, alphabet=alphabet, **IDENTIFY_PARAMS))


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
    llm = run_llm(positive, negative, 
                  desc=desc, verbose=verbose,
                  params=llm_params,
                  allow_unsure=allow_unsure,
                  llm_query_call_back=llm_query_call_back,
                  endpoint=llm_endpoint)
    next(llm)

    @cache
    def label(word):
        print(word)
        if word in positive:
            return True
        elif word in negative:
            return False
        return llm.send(word)


    # 2. Augment labeled examples with labeled examples.
    search = random_search if use_random_search else enumerative_search
    search = partial(search, alphabet=alphabet, max_depth=ce_search_depth)
    if use_random_search:
        search = partial(search, samples=random_iters)

    positive, negative = set(positive), set(negative)
    for word in search():
        bucket = positive if label(word) else negative
        bucket.add(word)
    positive, negative = list(positive), list(negative)

    # 3. Run learner.
    learner = guess_dfa_sat if use_dfa_identify else guess_dfa_lstar
    return learner(positive, negative, set(alphabet), label, active_queries)


if __name__ == "__main__":
    alphabet = ["red", "yellow", "blue", "brown"]
    positive = {("yellow",), ("yellow", "blue"), ("blue","brown","yellow")}
    negative = {("blue",), ("blue","yellow"), ("red", "blue", "red", "brown", "red", "brown")}
    desc = """
A robot is operating in a grid world and can visit four types of tiles:
{red, yellow, blue, brown}.

The robot is to visit tiles according to some set of rules.

Rules include:

    1. You need to reach a yellow tile by the end of the episode.
    2. Visiting a red tile results in failure.
    3. There is no penalty for visiting a color multiple times in a row.
    4. <unknown>

    """
    lang = guess_dfa(positive, negative,
                     desc=desc, hypothesize_rule=False,
                     verbose=True,
                     ce_search_depth=3,
                     random_iters=3,
                     active_queries=3,
                     alphabet=alphabet,
                     use_random_search=True,
                     allow_unsure=True,
                     use_dfa_identify=True)
    print(lang)
