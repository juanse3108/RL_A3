import time
import numpy as np

from reinforce import train_reinforce
from actor_critic import train_actor_critic
from a2c import train_a2c
from utils import smooth, load_baseline, LearningCurvePlot


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

N_REPETITIONS = 5
N_STEPS = 1000000
EVAL_INTERVAL = 2500
N_EVAL_EPISODES = 5
SMOOTHING = 9


# N_REPETITIONS = 1
# N_STEPS = 200000
# EVAL_INTERVAL = 2500
# SMOOTHING = 5

COMMON_PARAMS = dict(
    n_steps=N_STEPS,
    eval_interval=EVAL_INTERVAL,
    n_eval_episodes=N_EVAL_EPISODES,
    gamma=0.99,
    hidden_size=64
)


# ============================================================
# AVERAGE OVER REPETITIONS
# ============================================================

def average_over_repetitions(train_function, n_repetitions=5, smoothing_window=5, **kwargs):
    """
    Run one algorithm several times and average the learning curves.

    Parameters
    ----------
    train_function : callable
        Training function to run, e.g. train_reinforce.
    n_repetitions : int
        Number of independent runs.
    smoothing_window : int
        Moving average window for smoothing curves.
    kwargs : dict
        Hyperparameters passed to the training function.

    Returns
    -------
    mean_curve : np.ndarray
        Average smoothed return curve.
    steps : np.ndarray
        Evaluation steps.
    """

    all_returns = []
    start_time = time.time()

    for rep in range(n_repetitions):
        print(f"  Repetition {rep + 1}/{n_repetitions}")
        returns, steps = train_function(**kwargs)
        all_returns.append(returns)

    min_len = min(len(r) for r in all_returns)
    all_returns = np.array([r[:min_len] for r in all_returns])
    steps = steps[:min_len]

    mean_curve = np.mean(all_returns, axis=0)

    if smoothing_window is not None:
        mean_curve = smooth(mean_curve, smoothing_window)
        steps = steps[:len(mean_curve)]

    print(f"  Finished in {(time.time() - start_time) / 60:.2f} minutes")

    return mean_curve, steps


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    """
    Run REINFORCE, Actor-Critic, and A2C and plot comparison.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    print("Running REINFORCE")
    reinforce_curve, reinforce_steps = average_over_repetitions(
        train_reinforce,
        n_repetitions=N_REPETITIONS,
        smoothing_window=SMOOTHING,
        learning_rate=1e-3,
        **COMMON_PARAMS
    )

    print("\nRunning Actor-Critic")
    ac_curve, ac_steps = average_over_repetitions(
        train_actor_critic,
        n_repetitions=N_REPETITIONS,
        smoothing_window=SMOOTHING,
        actor_lr=1e-4,
        critic_lr=1e-3,
        **COMMON_PARAMS
    )

    print("\nRunning A2C")
    a2c_curve, a2c_steps = average_over_repetitions(
        train_a2c,
        n_repetitions=N_REPETITIONS,
        smoothing_window=SMOOTHING,
        actor_lr=1e-3,
        critic_lr=1e-3,
        **COMMON_PARAMS
    )

    plot = LearningCurvePlot(
        title="REINFORCE vs Actor-Critic vs A2C on CartPole"
    )
    plot.set_ylim(0, 520)

    plot.add_curve(reinforce_steps, reinforce_curve, label="REINFORCE")
    plot.add_curve(ac_steps, ac_curve, label="Actor-Critic")
    plot.add_curve(a2c_steps, a2c_curve, label="A2C")

    try:
        baseline_steps, baseline_returns = load_baseline()
        plot.add_curve(baseline_steps, baseline_returns, label="Assignment 2 baseline")
    except FileNotFoundError:
        print("BaselineDataCartPole.csv not found. Plotting without baseline.")

    plot.save("results/comparison.png")


if __name__ == "__main__":
    main()