import os

from lstar_lm.llm import DEFAULT_ENDPOINT
from lstar_lm import guess_dfa


def test_api():
    endpoint = os.environ.get('ENDPOINT', DEFAULT_ENDPOINT)

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
                     task_description=desc,
                     verbose=True,
                     ce_search_depth=3,
                     random_iters=3,
                     active_queries=3,
                     alphabet=alphabet,
                     use_random_search=True,
                     allow_unsure=True,
                     use_dfa_identify=False,
                     llm_endpoint=endpoint)
    print(lang)


if __name__ == '__main__':
    test_api()
