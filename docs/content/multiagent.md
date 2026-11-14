# Multi-agent environments

This module brings the task families from [marlgrid](https://github.com/kandouss/marlgrid)
to current MiniGrid objects. It addresses [#514](https://github.com/Farama-Foundation/Minigrid/issues/514)
through an optional, separate API. No existing single-agent environment, registration,
wrapper, object-code table or default dependency is replaced.

## Install and use

```bash
pip install -e ".[multiagent]"
```

PettingZoo Parallel is used because each environment accepts one joint action
per round and returns observations, rewards and completion flags per agent.
Inside a round, actions resolve in a seeded random order, as in marlgrid; they
are not simultaneous physics updates. PettingZoo's `parallel_to_aec` can expose
the same environments through its turn-oriented API.

```python
from minigrid.multiagent import EmptyParallelEnv

env = EmptyParallelEnv(num_agents=2, max_steps=100)
observations, infos = env.reset(seed=42)
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Exactly one valid action is required per live agent. Completed identities receive
one final observation and must be omitted from the next action dictionary. The
seven action numbers retain MiniGrid's meaning; action 6 is a no-op, not a request
to terminate. `restrict_actions=True` exposes only left, right and forward.

## Task families

| Class | Task and defaults |
| --- | --- |
| `EmptyParallelEnv` | Shared bottom-right goal, size 9, two agents. |
| `ClutteredParallelEnv` | Random wall clutter, size 11, three agents, density 0.15. |
| `DoorKeyParallelEnv` | A shared yellow key opens a locked door in a dividing wall. |
| `GoalCycleParallelEnv` | Visit yellow bonus tiles in cyclic order; size 13, density 0.1, three bonus tiles. |
| `VisibilityParallelEnv` | A dividing wall for observation/occlusion checks; no reward or terminal goal. |

Each agent has its own position, orientation, inventory and task state. The
current limit is six agents with distinct MiniGrid colors.

Defaults preserve the source navigation rules: `ghost_mode=True` permits overlap;
the horizon is 100 rounds; goal reward is `1 - 0.9 * step_count / max_steps`.
Use `reward_decay=False` for a constant goal reward of 1. `ghost_mode=False`
blocks movement into peers on empty tiles, but passable world objects can still
contain several agents, as in the source. One agent finishing does not end its peers.

Clutter can be specified by `n_clutter` or `clutter_density`, not both. Density
is multiplied by interior area and rounded down. `randomize_goal=True` selects
a random goal before clutter placement in Cluttered. Default spawn sampling
does not guarantee reachability, and can start on a goal without awarding a reward.
For Empty/Cluttered, `ensure_reachable=True` selects reachable non-goal starts;
it deliberately changes the map/start distribution.

DoorKey retains source starts on either side of the door. Set
`start_in_left_room=True` to require agents to begin on the key side. Only the
agent carrying a matching key can unlock the door; once open, peers can use it.

GoalCycle has independent `bonus_state` per agent. The first visited tile sets
the starting point; subsequent rewards require the next tile modulo
`n_bonus_tiles`. Repeated and out-of-order visits yield `-abs(penalty)`.
`initial_reward=False` suppresses the first reward; `reset_on_mistake=True`
uses a wrong visit as the new cycle position. Default reward is 1, penalty 0,
and reward decay is disabled. Bonus visits do not terminate agents. The source's
one-tile edge case is retained: it counts as a repeat, not a successful cycle.
Bonus tiles use the existing floor encoding with the bonus index in its state
channel; no global MiniGrid object IDs are added.

## Activation, respawning and observations

`spawn_delay` accepts an integer or one integer per agent. A waiting identity
remains in `env.agents`, returns a masked observation and ignores its action.
It activates before the first round whose previous `step_count` meets its delay.
This keeps Parallel API action keys predictable even if every agent is waiting.
`infos[agent]["active"]` identifies whether the agent is on the grid.

With `respawn=True`, a goal/lava arrival awards the transition reward and respawns
the same identity after all actions in the round. It is not reported as a
termination followed by resurrection. `infos[agent]["respawned"]` marks this event.
Inventory is cleared; direction, prestige and bonus-cycle state persist until
the next episode reset. Time limits still truncate the episode.

Prestige accumulates rewards and decays each active round by `prestige_beta`
(default 0.95). A negative reward resets it to zero unless
`allow_negative_prestige=True`. It is available in agent state and step info;
prestige-based color rendering is not implemented.

Observation formats are selected explicitly:

- `observation_style="symbolic"` (default): MiniGrid `image`, `direction`, `mission`.
- `observation_style="image"`: egocentric RGB, default 7 tiles at 5 pixels per tile.
- `observation_style="rich"`: `pov` RGB plus optional `reward`, normalized
  `position`, and `orientation`, enabled by the corresponding `observe_*` flags.

`agent_view_size`, `view_tile_size`, `view_offset`, `see_through_walls` and
`hide_item_types` configure the view. These settings are uniform across agents.
Hiding objects affects observations only, not physics. Inactive agents receive
masked images. `render_mode="rgb_array"` provides a global frame; human-window
rendering is not included. Existing single-agent Gymnasium wrappers should not
be applied directly to these multi-agent environments.

## Legacy calling convention

```python
from minigrid.multiagent import EmptyParallelEnv, MarlgridListAdapter

env = MarlgridListAdapter(EmptyParallelEnv(observation_style="image"))
observations = env.reset(seed=42)  # list in possible_agents order
observations, rewards, done, info = env.step([2, 0])
env.close()
```

The adapter keeps fixed slots, including completed agents, and supplies zero
rewards for inactive slots. It exposes Tuple spaces, validates joint actions
before advancing, and preserves detailed termination/truncation flags in `info`.
`seed(n)` defers reseeding to the next reset. It wraps the same physics engine;
it does not create a second implementation or change native `step` signatures.

`make_marlgrid(name, **overrides)` exposes the seven preset names from the source,
with RGB observations and the list adapter. For example:

```python
from minigrid.multiagent import make_marlgrid

env = make_marlgrid("MarlGrid-2AgentEmpty9x9-v0")
observations = env.reset(seed=42)
observations, rewards, done, info = env.step([2, 0])
env.close()
```

This is explicit call-shape compatibility, not a drop-in `marlgrid` package.
Change the construction/import entry point when migrating old scripts; legacy
names are not added to `gym.make()`. Source quirks such as the
`MarlGrid-1AgentCluttered15x15-v0` preset actually using size 11 are retained.

Reset restores east-facing headings instead of inheriting the previous episode's
headings. Colors, pixel rendering, visibility implementation and historical Gym
random-number sequences are not identical. Rich reward observations contain the
actual transition reward (the pinned source leaves that field at zero). Legacy
per-agent interface objects, custom color palettes and old checkpoint pixel
compatibility are not provided. The unused source `cycle_reset` argument is not
accepted silently. These differences do not affect existing MiniGrid users.

## Validation and source references

```bash
python -m pytest tests/test_multiagent.py tests/test_multiagent_tasks.py -q
python -m pytest tests -q
```

Tests exercise all five families with the Parallel API, seeding and AEC
conversion, then check joint actions, key contention, inventories, cycle reward
state, respawning, delayed activation, observation spaces and list/native parity.
Single-agent imports without PettingZoo, Gymnasium registry isolation and shared
render-cache behavior have dedicated checks.

The frozen fixtures record 22 cases / 194 joint steps from original
[marlgrid revision e88c40b](https://github.com/kandouss/marlgrid/tree/e88c40bad07653575ac11fe2f3a115e4de3d13e9).
They cover Empty, Cluttered, GoalCycle, visibility, respawning and delayed
activation. CI consumes JSON only and does not fetch or execute legacy code.
DoorKey has separate layout/interaction/end-to-end tests: the pinned source
cannot construct it because its `_rand_int` helper is missing. The negative
prestige path also references an undefined `rew` attribute in that source;
this implementation maintains the numeric prestige state instead.

To reproduce fixtures, clone that revision outside the project, install this
project's test/multiagent extras, and set `MARLGRID_SOURCE` to the clone path.
Run `scripts/marlgrid_reference/capture_navigation.py` and
`scripts/marlgrid_reference/capture_extended.py`; each prints JSON to stdout.
Only execute reference scripts in a separate process: their runtime adapts Gym
imports, old NumPy aliases, rendering helpers and the RNG method name. It does
not replace transition/reward methods. Visibility uses the source's
`see_through_walls=True` option because its old JIT-only occlusion routine is
not portable to this headless runtime. These are state-transition comparisons,
not pixel-parity claims or native legacy-dependency installation tests.

No learning algorithm, trained policy or GPU is needed for these checks.
