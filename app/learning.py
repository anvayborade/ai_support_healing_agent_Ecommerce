import random
from .storage import get_learning_params, update_learning

def thompson_rank_actions():
    params = get_learning_params()
    samples = []
    for a, ab in params.items():
        alpha, beta = ab["alpha"], ab["beta"]
        x = random.gammavariate(alpha, 1.0)
        y = random.gammavariate(beta, 1.0)
        p = x / (x + y + 1e-9)
        samples.append((a, p))
    samples.sort(key=lambda t: t[1], reverse=True)
    return samples

def record_outcome(action_type: str, success: bool):
    update_learning(action_type, success)