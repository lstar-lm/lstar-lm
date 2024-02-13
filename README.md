# L*LM

[![PyPI version](https://badge.fury.io/py/lstar_lm.svg)](https://badge.fury.io/py/lstar_lm)

Implementation of L*LM algorithm algorithm. See [project
page](http://lstar-lm.github.io) for details.


**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)


# Installation

If you just need to use `lstar_lm`, you can just run:

`$ pip install lstar_lm`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

The main entry point for using this library is the `guess_dfa` function.

```python
from lstar_lm import guess_dfa
```

An invocation of `guess_dfa` takes the form.
```python


dfa = guess_dfa(
    positive = ...,  # List of positive examples. Each example is a list of tuples of tokens.
    negative = ...,  # List of negative examples. Each example is a list of tuples of tokens.
    alphabet = ...,  # List of (hashable) tokens.
    task_description = ...,  # String of task description.
    allow_unsure = ...,      # Whether to allow unsure responses (default True).
    random_iters = ...,      # Number of random queries to oracle.
    active_queries = ...,    # Number of active queries to oracle.
    use_dfa_identify = ...,  # True if use SAT based DFA identification. False uses L* + SAT hybrid.
    llm_endpoint = ...,      # http endpoint for llama.cpp server (default "http://localhost:8080/completion").
)
```
