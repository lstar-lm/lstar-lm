import httpx
import lstar
import re
from lstar_llm.common import DEFAULT_PARAMS
from lstar.learn import _learn_dfa, learn_dfa
from lstar import iterative_deeping_ce

ENDPOINT = "http://127.0.0.1:8080/completion"

BASE_PROMPT = """
Tomita Grammar is a language that can only be made up of "0" and "1".

Your task is to answer the following questions using step-by-step reasoning to
show your work. Do not answer any other question. When you arrive at a
conclusion, please state it as FINAL_ANSWER: <yes, no>
"""

MEMBERSHIP_GRAMMAR = """
  root ::= work answer
  work ::= [^"FINAL_ANSWER"]+
  answer ::= "FINAL_ANSWER: " ("yes" | "no")
"""

def tomita_1(word):
    return not "0" in word

def tomita_1_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the following word contain a 0: {word}?\n"
    
    return prompt

def tomita_1_ce(dfa):
    count = len(dfa.states())
    if count == 2: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "1"*(count//2)

def tomita_2(word):
    return word=="10"*(int(len(word)/2))

def tomita_2_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f'Does the following word repeat EXACTLY as alternating "0" and "1" \
        without breaking the pattern? In other words, is every 0 followed by \
        a 1 and every 1 followed by a 0? {word}?\n'
    
    return prompt

def tomita_2_ce(dfa):
    count = len(dfa.states())
    if count == 3: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "10"*(count//2)

_not_tomita_3 = re.compile("((0|1)*0)*1(11)*(0(0|1)*1)*0(00)*(1(0|1)*)*$") 
# *not* tomita 3: words containing an odd series of consecutive ones and then later an odd series of consecutive zeros
# tomita 3: opposite of that
def tomita_3(w): 
    return None is _not_tomita_3.match(w) #complement of _not_tomita_3

def tomita_3_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the following word contain an odd consecutive series of 0's \
        followed by an odd consecutive series of 1's: {word}?\n"
    
    return prompt

def tomita_3_ce(dfa):
    count = len(dfa.states())
    if count == 5: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "0"*(count//2) + "1"*(count//2)

def tomita_4(word):
    return not "000" in word

def tomita_4_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the following word contain a exactly 3 consecutive '0': {word}?\n"
    
    return prompt

def tomita_4_ce(dfa):
    count = len(dfa.states())
    if count == 4: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "000"*(count//3)

def tomita_5(word):
    return (word.count("0")%2 == 0) and (word.count("1")%2 == 0)

def tomita_5_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the following word contain an even number of 0's and an even number of 1's: {word}?\n"
    
    return prompt

def tomita_5_ce(dfa):
    count = len(dfa.states())
    if count == 4: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        even_count = count//2 + 1 if count%2 == 1 else count//2
        return "0"*even_count + "1"*even_count

def tomita_6(word):
    return ((word.count("0")-word.count("1"))%3) == 0

def tomita_6_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the number of 0's minus the number of 1's divide by 3 evenly: {word}?\n"
    
    return prompt

def tomita_6_ce(dfa):
    count = len(dfa.states())
    if count == 3: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "0"*(count//3) + "1"*(count//3)

def tomita_7(word):
    return word.count("10") <= 1

def tomita_7_llm(word):
    prompt = BASE_PROMPT + "\n\n" + \
        f"Does the following word contain at most one occurrence of '10': {word}?\n"
    
    return prompt

def tomita_7_ce(dfa):
    count = len(dfa.states())
    if count == 5: # minimal DFA
        return None
    else:
        # I need to return a counterexample
        return "1010"*(count//5)



def query_llm(word, tomita_query):
    prompt = tomita_query(word) + "\nAI:\n"

    data = DEFAULT_PARAMS | {"prompt": prompt, "grammar": MEMBERSHIP_GRAMMAR}

    response = httpx.post(ENDPOINT, json=data).json()

    print("Response from LLM Teacher:\n" + response["content"])

    content = response["content"]

    return content


def test_tomita_1():
    dfa = learn_dfa(
        inputs={0, 1},
        label=lambda word: tomita_1_llm(word),
        find_counter_example=iterative_deeping_ce(tomita_1, depth=7),
        outputs=None
    )

    return dfa

def test_tomita_2():
    dfa = learn_dfa(
        inputs={0, 1},
        label=lambda word: tomita_2_llm(word),
        find_counter_example=lambda word: iterative_deeping_ce(tomita_2_ce(word), depth=7),
        outputs={0, 1, 2},
    )

    return dfa

print(test_tomita_1())
# print(test_tomita_2())
