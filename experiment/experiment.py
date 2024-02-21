# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# %%
import os
from dataclasses import dataclass
from functools import partial
from pprint import pprint
from typing import Any

import dfa
from diss import diss, LabeledExamples
from diss.concept_classes.dfa_concept import DFAConcept
from diss.experiment.planner import GridWorldPlanner

from lstar_lm import dfa_search_with_diss, DEFAULT_ENDPOINT

# %%
PROMPT = """
A robot is operating in a grid world and can visit four types of tiles:
{red, yellow, blue, green}.

They correspond to lava (red), recharging (yellow), water (blue), and drying (green) tiles.

The robot is to visit tiles according to some set of rules. This will be recorded as a
sequence of colors.

Rules include:

    1. The sequence must contain at least one yellow tile, i.e., eventually recharge. 
    2. The sequence must not contain any red tiles, i.e., lava must be avoided at all costs.
    3. If blue is visited, then you must visit green *before* yellow, i.e., the robot must dry off before recharging.

A positive example must conform to all rules.

Further note that repeated sequential colors can be replaced with a single instance.

For example:
 - [yellow,yellow,blue] => [yellow, blue]
 - [red,red,blue,green,green,red] => [red,blue,green,red]
 - [blue,blue,blue] => [blue]
"""

# %%
planner = GridWorldPlanner.from_string(
    buff="""y....g..
    ........
    .b.b...r
    .b.b...r
    .b.b....
    .b.b....
    rrrrrr.r
    g.y.....""",
    start=(3, 5),
    slip_prob=1/32,
    horizon=15,
    policy_cache='diss_experiment.shelve',
)

to_demo = planner.to_demo

def to_chain(c, t, psat):
    return planner.plan(c, t, psat, monolithic=True, use_rationality=True)


# %%
TRC4 = [
    (3, 5),
    {'a': '↑', 'c': 0},
    {'a': '↑', 'c': 1},
    {'a': '↑', 'c': 1},
    {'a': '→', 'c': 1},
    {'a': '↑', 'c': 1},
    {'a': '↑', 'c': 1},
    {'a': '→', 'c': 1},
    {'a': '→', 'c': 1},
    {'a': '→', 'c': 1},
    {'a': '←', 'c': 1},
    {'a': '←', 'c': 1},
    {'a': '←', 'c': 1},
    {'a': '←', 'c': 1},
    {'a': '←', 'c': 1, 'EOE_ego': 1},
]

TRC5 = [
    (3, 5),
    {'a': '↑', 'c': 1},
    {'a': '↑', 'c': 1},
    {'a': '↑', 'c': 1},
    {'a': '←', 'c': 1},
    {'a': '←', 'c': 1, 'EOE_ego': 1},
]


# Binary encode demonstrations for BDD based planner.
demos = [to_demo(TRC4), to_demo(TRC5)]
path_to_colors = planner.lift_path


# %%
def analyze(search):
    concept2energy = {}    # Explored concepts + associated energy

    # Run Search and collect concepts, energy, and POI.
    for data, concept, metadata in search:
        concept2energy[concept] = metadata['energy']

    return sorted(list(concept2energy), key=concept2energy.get)

endpoint = os.environ.get('ENDPOINT', DEFAULT_ENDPOINT)
search = dfa_search_with_diss(alphabet={"red", "yellow", "blue", "green"},
                              # LLM Oracle params
                              task_description=PROMPT,
                              llm_endpoint=endpoint,
                              allow_unsure=True,
                              # Active dfa learner params
                              ce_search_depth=3,
                              active_queries=3,
                              use_dfa_identify=True,
                              # DISS params
                              to_chain = to_chain,
                              demonstrations = demos,
                              max_diss_iters=3,
                              diss_params={
                                  "sgs_temp": 2**-7,
                                  "surprise_weight": 1,
                                  "reset_period": 30,
                                  "size_weight": 1/80,
                                  "example_drop_prob": 1/20,
                                  "synth_timeout": 0,
                                  "lift_path": path_to_colors,
                             })
pprint(analyze(search))
