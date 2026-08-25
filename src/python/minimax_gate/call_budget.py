import os

MAX_CALLS = 4
calls_made = 0
cost_per_call = 0.0035

def check_budget():
    global calls_made
    if calls_made >= MAX_CALLS:
        raise RuntimeError(f"API budget exceeded. Max {MAX_CALLS} calls allowed.")
    calls_made += 1

def get_call_count():
    global calls_made
    return calls_made

def get_estimated_cost():
    global calls_made
    return calls_made * cost_per_call

def reset_budget():
    global calls_made
    calls_made = 0
