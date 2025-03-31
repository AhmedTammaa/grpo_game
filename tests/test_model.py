import pytest
import jax
import jax.numpy as jnp
import time
from neural_network.model import init_params, forward
from neural_network.utils import get_input
from constants.physics import M_total


@pytest.fixture
def network_setup():
    key = jax.random.PRNGKey(42)  # Fixed seed for testing
    hidden_size = 32
    input_size = 13  # This should match your actual input size
    output_size = 56  # 64-8 as per requirement

    params = init_params(key, input_size, hidden_size, output_size)

    my_velocities_init = jnp.zeros((3, 2))
    my_masses_init = jnp.array([1.0, 1.0, 1.0]) * (M_total) / 3.0
    my_points_init = jnp.array(
        [
            [-1.0, 0.0],  # First vertex
            [1.0, 0.0],  # Second vertex
            [0.0, 1.732],  # Third vertex
        ]
    )

    return {
        "params": params,
        "velocities": my_velocities_init,
        "masses": my_masses_init,
        "points": my_points_init,
    }


def test_network_forward(network_setup):
    # Test data
    my_muscles = jnp.array([0.0, 1.0, 0.0])
    my_pumps = jnp.array([1.0, 1.0, 0.0])

    # Get network components from fixture
    params = network_setup["params"]
    my_points = network_setup["points"]
    my_velocities = network_setup["velocities"]
    my_masses = network_setup["masses"]

    # Get inputs and run forward pass
    inputs = get_input(
        my_points, my_velocities, my_masses, my_pumps, my_muscles
    )
    log_probs = forward(params, inputs)

    # Assertions
    assert log_probs.shape == (56,)  # Check output shape
    assert jnp.isfinite(log_probs).all()  # Check for any NaN or inf values
    assert (
        jnp.abs(jnp.sum(jnp.exp(log_probs)) - 1.0) < 1e-6
    )  # Should sum to approximately 1
