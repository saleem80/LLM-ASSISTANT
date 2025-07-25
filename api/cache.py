import hashlib
import json

# Temporary in-memory cache
caches = {}

def make_cache_key(context, question):
    unique_string = f"{context}:{question}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def get_cached_answer(context, question):
    key = make_cache_key(context, question)
    cached = caches.get(key)
    if cached:
        return json.loads(cached) 
    return None

def set_cached_answer(context, question, answer):
    key = make_cache_key(context, question)
    caches[key] = json.dumps(answer)
