from functools import cache

import httpx


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


DEFAULT_ENDPOINT ="http://127.0.0.1:8080/completion"


BASE_PROMPT = """
{task_description}

Additionally, by examining demonstrations of the task, we conjecture the following labeled examples:

"""

MEMBERSHIP_PROMPT = """
Please briefly answer the following questions using step-by-step reasoning to
show your work. Do not answer any other question. When you arrive at a
conclusion, please state it as:\n\n
Answer: <yes, no>
"""

DELIM = "Answer: "

MEMBERSHIP_GRAMMAR = f"""
  root ::= work delim answer
  delim ::= "{DELIM}"
  work ::= [^"{DELIM}"]+
  answer ::= ("yes" | "no")
"""

MEMBERSHIP_PROMPT_IDK = """
Please briefly answer the following questions using step-by-step reasoning to
show your work. Do not answer any other question. When you arrive at a
conclusion, please state it as\n\n
Answer: <yes, no, unsure>
"""

MEMBERSHIP_GRAMMAR_IDK = f"""
  root ::= work delim answer
  delim ::= "{DELIM}"
  work ::= [^"{DELIM}"]+
  answer ::= ("yes" | "no" | "unsure")
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
    match msg.split(DELIM):
        case [_, 'yes']:    return True
        case [_, 'unsure']: return None
        case _:             return False


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
        response = httpx.post(endpoint, timeout=1000, json=data).json()
        content = response["content"]
        llm_query_call_back(prompt, content)
        prompt += f"{content}\nUser:"

        if verbose: print(query + f"{content}\nUser:")
        word = yield parse_membership_reponse(content)


def llm_oracle(positive=(),
               negative=(),
               desc="",
               verbose=True,
               params=DEFAULT_PARAMS,
               allow_unsure=True,
               llm_query_call_back=lambda *_: None,
               endpoint=DEFAULT_ENDPOINT):
    llm = run_llm(positive, negative, 
                  desc=desc, verbose=verbose,
                  params=params,
                  allow_unsure=allow_unsure,
                  llm_query_call_back=llm_query_call_back,
                  endpoint=endpoint)
    next(llm)

    @cache
    def label(word):
        print(word)
        if word in positive:
            return True
        elif word in negative:
            return False
        return llm.send(word)

    return label


__all__ = [ 'DEFAULT_PARAMS', 'DEFAULT_ENDPOINT', 'run_llm' ]
