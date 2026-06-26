"""UCB bandit agent for action selection.

State  : (player_id, side) — e.g. "CENTER/middle_right"
Actions: named entries in play["actions"] — e.g. "v0", "v1"

Q-table entry: {"reward": float, "trials": int}
  reward — running mean of observed rewards
  trials — number of times this action was selected

Warm-start: calibrated actions ("v0") initialize with reward=1.0, trials=1
so they are preferred early but still compete with new variants over time.

UCB score = mean_reward + C * sqrt(log(total_trials) / trials)
A higher C explores more; lower C exploits more.
"""

import json
import math
import os

from engine.constants import PlayerID

_Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.json")
_WARM_START_REWARD = 1.0
_WARM_START_TRIALS = 1
_UCB_C = 1.0  # exploration constant

_play_mode = False


def set_play_mode(enabled: bool) -> None:
    """Play mode disables exploration — always picks the highest mean reward action."""
    global _play_mode
    _play_mode = enabled
    print(f"RL: {'play' if enabled else 'training'} mode")


def _load() -> dict:
    if os.path.exists(_Q_TABLE_PATH):
        with open(_Q_TABLE_PATH) as f:
            return json.load(f)
    return {}


def _save(table: dict) -> None:
    with open(_Q_TABLE_PATH, "w") as f:
        json.dump(table, f, indent=2)


_table = _load()


def _state_key(player_id: PlayerID, side: str) -> str:
    return f"{player_id.name}/{side}"


def _ucb_score(entry: dict, total_trials: int) -> float:
    return entry["reward"] + _UCB_C * math.sqrt(math.log(total_trials) / entry["trials"])


def select_action(play: dict, player_id: PlayerID, side: str) -> dict:
    """Select an action from play["actions"] using UCB. Returns the chosen action dict."""
    key = _state_key(player_id, side)
    state = _table.setdefault(key, {})

    for action in play["actions"]:
        name = action["name"]
        if name not in state:
            state[name] = {"reward": _WARM_START_REWARD, "trials": _WARM_START_TRIALS}

    if _play_mode:
        chosen_name = max(state, key=lambda n: state[n]["reward"])
        print(f"RL: selected '{chosen_name}' for '{key}'  (play mode)")
    else:
        total_trials = sum(v["trials"] for v in state.values())
        chosen_name = max(state, key=lambda n: _ucb_score(state[n], total_trials))
        scores = {n: round(_ucb_score(v, total_trials), 3) for n, v in state.items()}
        print(f"RL: selected '{chosen_name}' for '{key}'  scores={scores}")

    return next(a for a in play["actions"] if a["name"] == chosen_name)


def update(player_id: PlayerID, side: str, action_name: str, reward: float) -> None:
    """Update the Q-table with the observed reward and persist to disk."""
    if action_name is None:
        return
    key = _state_key(player_id, side)
    state = _table.setdefault(key, {})
    if action_name not in state:
        state[action_name] = {"reward": _WARM_START_REWARD, "trials": _WARM_START_TRIALS}

    entry = state[action_name]
    entry["trials"] += 1
    entry["reward"] += (reward - entry["reward"]) / entry["trials"]
    _save(_table)
    print(f"RL: updated '{key}/{action_name}': mean_reward={entry['reward']:.3f}, trials={entry['trials']}")
