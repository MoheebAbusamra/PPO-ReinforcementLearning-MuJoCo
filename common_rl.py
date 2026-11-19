"""
common_rl.py  -  SHARED TEAM FILE (ANN 26CSCI33H, RL track)
Used unchanged by MBPO, SAC, TD3 and PPO.
If something must change, change it for everyone and re-share.

Gives every model the same:
  - settings (environments, budgets, seeds, gamma, evaluation rule)
  - environment creation + seeding
  - test evaluation (deterministic policy, separate env)
  - TensorBoard tags, folder names, final score and results table
"""
import os, json, time, glob, random, platform
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------
# 1. SHARED SETTINGS  (team agreement, items 1-8)
# ---------------------------------------------------------------
ENVS = {
    #  env id              training budget (env steps)   test every N steps
    "InvertedPendulum-v5": {"total_steps": 50_000,  "eval_every": 2_000},
    "Hopper-v5":           {"total_steps": 300_000, "eval_every": 10_000},
    "Walker2d-v5":         {"total_steps": 300_000, "eval_every": 10_000},
    "HalfCheetah-v5":      {"total_steps": 300_000, "eval_every": 10_000},
}
SEEDS = [0, 1, 2]
GAMMA = 0.99                 # discount factor, used by all four models
EVAL_EPISODES = 10           # episodes per test evaluation
EVAL_SEED_OFFSET = 10_000    # test env uses different seeds from the training env
FINAL_SCORE_LAST_N = 3       # final score = mean of the last 3 test evaluations
LOG_ROOT = "runs"            # runs/{env}/{algo}/{spec}/seed{n}


# ---------------------------------------------------------------
# 2. SEEDING
# ---------------------------------------------------------------
def set_seed(seed: int):
    """Seed python, numpy and torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------
# 3. ENVIRONMENTS
# ---------------------------------------------------------------
def make_env(env_id: str, seed: int):
    """Training env with Gymnasium defaults + episode statistics.
    Add your own model-specific wrappers ON TOP of this
    (e.g. PPO: ClipAction, NormalizeObservation, NormalizeReward)."""
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_env_fn(env_id: str, seed: int):
    """Same as make_env but returns a function, for vector envs:
    gym.vector.SyncVectorEnv([make_env_fn(env_id, seed + i) for i in range(n)])"""
    return lambda: make_env(env_id, seed)


# ---------------------------------------------------------------
# 4. TEST EVALUATION  (identical for all models)
# ---------------------------------------------------------------
def evaluate(act_fn, env_id: str, seed: int, n_episodes: int = EVAL_EPISODES):
    """Run n_episodes on a separate, unwrapped test env.

    act_fn(obs) -> action : YOUR model's DETERMINISTIC action for ONE raw observation
        PPO : mean of the Gaussian 
        SAC : tanh(mean), scaled to the action range
        TD3 : actor output, no exploration noise
        MBPO: its SAC agent's deterministic action
    Returns (mean_return, std_return, list_of_returns)."""
    env = gym.make(env_id)
    low, high = env.action_space.low, env.action_space.high
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + EVAL_SEED_OFFSET + ep)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                action = np.asarray(act_fn(obs), dtype=np.float32).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, _ = env.step(np.clip(action, low, high))
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


# ---------------------------------------------------------------
# 5. LOGGER  (shared TensorBoard tags + folders + final score)
# ---------------------------------------------------------------
class RunLogger:
    """One per run. Usage:
        logger = RunLogger("PPO", "baseline", "Hopper-v5", seed=0, hparams={...})
        logger.log_train_episodes(step, infos)        # after every env.step
        logger.log_losses(step, actor_loss, critic_loss)
        logger.log(step, "ppo/approx_kl", value)      # your own extras
        logger.maybe_evaluate(step, act_fn)           # after every env.step / update
        logger.close()                                # at the end
    """

    def __init__(self, algo, spec, env_id, seed, hparams=None):
        self.algo, self.spec, self.env_id, self.seed = algo, spec, env_id, seed
        self.cfg = ENVS[env_id]
        self.dir = os.path.join(LOG_ROOT, env_id, algo, spec, f"seed{seed}")
        os.makedirs(self.dir, exist_ok=True)
        self.writer = SummaryWriter(self.dir)
        self.start = time.time()
        self.next_eval = self.cfg["eval_every"]
        self.evals = []  # (step, mean_return, std_return)
        self.hparams = hparams or {}
        self.writer.add_text("hparams", json.dumps(self.hparams, indent=2, default=str))
        self.writer.add_text("hardware", self._hardware())

    @staticmethod
    def _hardware():
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        return f"{platform.platform()} | CPU: {platform.processor() or platform.machine()} | GPU: {gpu} | torch {torch.__version__} | gymnasium {gym.__version__}"

    def log(self, step, tag, value):
        self.writer.add_scalar(tag, float(value), step)

    def log_losses(self, step, actor_loss=None, critic_loss=None):
        if actor_loss is not None:
            self.log(step, "losses/actor_loss", actor_loss)
        if critic_loss is not None:
            self.log(step, "losses/critic_loss", critic_loss)

    def log_train_episodes(self, step, infos):
        """Pass the `info`/`infos` from env.step() works for single and vector envs."""
        if "episode" not in infos:
            return
        ep = infos["episode"]
        if np.ndim(ep["r"]) == 0:                      # single env
            self.log(step, "train/episodic_return", ep["r"])
            self.log(step, "train/episodic_length", ep["l"])
        else:                                          # vector env
            mask = infos.get("_episode", np.ones_like(ep["r"], dtype=bool))
            for r, l in zip(ep["r"][mask], ep["l"][mask]):
                self.log(step, "train/episodic_return", r)
                self.log(step, "train/episodic_length", l)

    def maybe_evaluate(self, step, act_fn):
        """Runs the test evaluation whenever `step` passes the next eval point
        (works for step-by-step off-policy loops and big PPO jumps)."""
        ran = False
        while step >= self.next_eval:
            self.next_eval += self.cfg["eval_every"]
            ran = True
        if ran:
            mean, std, _ = evaluate(act_fn, self.env_id, self.seed)
            self.evals.append((step, mean, std))
            self.log(step, "test/mean_return", mean)
            self.log(step, "test/std_return", std)
            self.log(step, "time/wall_clock", time.time() - self.start)
            print(f"[{self.algo}|{self.spec}|{self.env_id}|seed{self.seed}] step {step}: test return {mean:.1f} ± {std:.1f}")
        return ran

    def final_score(self):
        last = [m for _, m, _ in self.evals[-FINAL_SCORE_LAST_N:]]
        return float(np.mean(last)) if last else float("nan")

    def close(self):
        wall = time.time() - self.start
        summary = {
            "algo": self.algo, "spec": self.spec, "env": self.env_id, "seed": self.seed,
            "final_score": self.final_score(), "wall_clock_sec": wall,
            "evals": self.evals, "hparams": self.hparams, "hardware": self._hardware(),
        }
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.writer.close()
        print(f"Done: final score {summary['final_score']:.1f}, {wall/60:.1f} min -> {self.dir}")


# ---------------------------------------------------------------
# 6. RESULTS TABLE  (team agreement, item 14)
# ---------------------------------------------------------------
def results_table(algo=None):
    """Reads every summary.json and prints a markdown table:
    final score mean ± std across seeds, and mean wall-clock minutes."""
    rows = {}
    for path in glob.glob(os.path.join(LOG_ROOT, "*", "*", "*", "seed*", "summary.json")):
        s = json.load(open(path))
        if algo and s["algo"] != algo:
            continue
        rows.setdefault((s["algo"], s["spec"], s["env"]), []).append(s)
    lines = ["| Algo | Spec | Env | Final score (mean ± std) | Seeds | Time (min) |",
             "|---|---|---|---|---|---|"]
    for (a, sp, e), runs in sorted(rows.items()):
        sc = [r["final_score"] for r in runs]
        t = np.mean([r["wall_clock_sec"] for r in runs]) / 60
        lines.append(f"| {a} | {sp} | {e} | {np.mean(sc):.1f} ± {np.std(sc):.1f} | {len(runs)} | {t:.1f} |")
    table = "\n".join(lines)
    print(table)
    return table
