
ENDPOINT = "http://127.0.0.1:8080/completion"

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
    "slot_id": 0,
    "stop": ["</s>", "AI:", "User:"],
    #"stream": False,
    "temperature": 0.1,
    "tfs_z": 1,
    "top_k": 40,
    "top_p": 0.5,
    "typical_p": 1,
}