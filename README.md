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

The main entry points for using this library are the `guess_dfa` and
`dfa_search_with_diss` functions.

- `guess_dfa` supports labeled examples and natural language.
- `dfa_search_with_diss` supports labeled examples, natural language, and
   demonstrations.

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

To learn using demonstrations, one can use `dfa_search_with_diss` to search for low energy DFAs:

```python
search = dfa_search_with_diss(
     alphabet = ..., # List of (hashable) tokens.

     # ---- Passive identification params ---
     positive = ..., # List of positive examples. Each example is a list of tuples of tokens.
     negative = ..., # List of negative examples. Each example is a list of tuples of tokens.

     # ---- LLM oracle params ----
     task_description = ...,              # String of task description.
     llm_params = ...,                    # Dictionary of llama.cpp parameters.
     llm_query_callback = ...,            # Callback sent prompt, response.
     llm_endpoint = ...,                  # http endpoint for llama.cpp server
                                          #    (default "http://localhost:8080/completion").
     allow_unsure = ...,                  # Whether to allow LLM to output unsure.

     # --- Active learning params ---
     random_iters = ...,                  # Number of random queries to oracle.
     active_queries = ...,                # Number of active queries to oracle.
     use_dfa_identify = ...,              # True if use SAT based DFA identification.
                                          # False uses L* + SAT hybrid.

     # --- Demonstration learning parameters ---
     demonstrations = ...,                # List of expert demonstrations
     max_diss_iters = ...,                # Maximum number of diss iterations.
     to_chain = ...,                      # Converted dfa concept to annotated markov chain given
                                          #   a maximum entropy policy. See DISS documentation.
     diss_params,                         # Other DISS parameters to override defaults.
)


mle = min(search, key=lambda _1, _2, metadata: metadata['energy'])
conjectured_examples, concept, metadata = mle
dfa = concept.dfa  # Most likely dfa
```
