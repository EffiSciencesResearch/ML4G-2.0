from typing import Iterable, Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.figure
import torch as t
import torch.nn.functional as F
from einops import rearrange, repeat
import torch
from tqdm.auto import tqdm
import numpy as np


def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def optimize_function(
    function: callable, parameters: t.Tensor, optimizer, n_steps: int
) -> List[t.Tensor]:
    trajectory = []
    for _ in range(n_steps):
        trajectory.append(parameters.detach().clone())
        loss = function(*parameters)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    trajectory = t.stack(trajectory).float()
    return trajectory


def test_SGD(optimizer_class):
    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    N_steps = 100
    learning_rate = 0.001
    optimizer = optimizer_class(parameters, learning_rate)

    test_trajectory = optimize_function(
        rosenbrocks_banana, parameters, optimizer, N_steps
    )

    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    solution_optimizer = t.optim.SGD([parameters], lr=learning_rate)
    solution_trajectory = optimize_function(
        rosenbrocks_banana, parameters, solution_optimizer, N_steps
    )

    assert t.allclose(test_trajectory, solution_trajectory, atol=1e-3)

    print("SGD test passed")


def test_momentum(optimizer_class):
    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    N_steps = 100
    learning_rate = 0.001
    optimizer = optimizer_class(parameters, learning_rate, beta=0.9)

    test_trajectory = optimize_function(
        rosenbrocks_banana, parameters, optimizer, N_steps
    )

    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    solution_optimizer = t.optim.SGD([parameters], lr=learning_rate, momentum=0.9)
    solution_trajectory = optimize_function(
        rosenbrocks_banana, parameters, solution_optimizer, N_steps
    )

    assert t.allclose(test_trajectory, solution_trajectory, atol=1e-3)

    print("Momentum test passed")


def test_RMSprop(optimizer_class):
    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    N_steps = 100
    learning_rate = 0.001
    epsilon = 1e-8
    optimizer = optimizer_class(parameters, learning_rate, beta=0.9, epsilon=epsilon)

    test_trajectory = optimize_function(
        rosenbrocks_banana, parameters, optimizer, N_steps
    )

    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    solution_optimizer = t.optim.RMSprop(
        [parameters], lr=learning_rate, alpha=0.9, eps=epsilon
    )
    solution_trajectory = optimize_function(
        rosenbrocks_banana, parameters, solution_optimizer, N_steps
    )

    assert t.allclose(test_trajectory, solution_trajectory, atol=1e-3)

    print("RMSprop test passed")


def test_Adam(optimizer_class):
    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    N_steps = 100
    learning_rate = 0.001
    epsilon = 1e-8
    optimizer = optimizer_class(
        parameters, learning_rate, beta1=0.9, beta2=0.999, epsilon=epsilon
    )

    test_trajectory = optimize_function(
        rosenbrocks_banana, parameters, optimizer, N_steps
    )

    parameters = t.tensor([-1.0, 2.0], requires_grad=True)
    solution_optimizer = t.optim.Adam(
        [parameters], lr=learning_rate, betas=(0.9, 0.999)
    )
    solution_trajectory = optimize_function(
        rosenbrocks_banana, parameters, solution_optimizer, N_steps
    )

    assert t.allclose(test_trajectory, solution_trajectory, atol=1e-3)

    print("Adam test passed")


def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


def plot_rosenbrock(
    trajectories={}, xmin=-2, xmax=2, ymin=-1, ymax=3, n_points=50
) -> matplotlib.figure.Figure:
    """Plot the rosenbrocks_banana function in 3D and its contour plot over the specified domain with trajectories."""

    global_minimum = t.tensor([1, 1])

    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection="3d")
    x = t.linspace(xmin, xmax, n_points)
    y = t.linspace(ymin, ymax, n_points)
    xx = repeat(x, "x -> y x", y=n_points)
    yy = repeat(y, "y -> y x", x=n_points)
    zs = rosenbrocks_banana(xx, yy)
    ax1.plot_surface(xx, yy, zs, cmap="viridis", alpha=0.5)

    for label, trajectory in trajectories.items():
        ax1.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            rosenbrocks_banana(trajectory[:, 0], trajectory[:, 1]),
            label=label,
            linewidth=4.0,
        )
    # plot the global minimum
    ax1.scatter(
        *global_minimum,
        rosenbrocks_banana(*global_minimum),
        color="red",
        label="Global Minimum"
    )

    ax1.set(xlabel="x", ylabel="y", zlabel="z")
    ax1.legend()

    # Contour plot
    ax2 = fig.add_subplot(122)
    zs = rosenbrocks_banana(xx, yy)

    levels = np.logspace(np.log10(zs.min()), np.log10(zs.max()), 10)
    contour = ax2.contour(x, y, zs, levels=levels, cmap="viridis")

    cbar = fig.colorbar(contour, ax=ax2)
    cbar.ax.set_ylabel("Function Value")

    for label, trajectory in trajectories.items():
        ax2.plot(trajectory[:, 0], trajectory[:, 1], label=label, linewidth=4.0)

    # plot the global minimum
    ax2.scatter(*global_minimum, color="red", label="Global Minimum")

    ax2.set(xlabel="x", ylabel="y")
    ax2.legend()

    plt.tight_layout()
    return fig
