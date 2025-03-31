import jax
import jax.numpy as jnp


def init_params(key, input_size, hidden_size, output_size):
    """Initialize network parameters (weights and biases)"""
    # Weights and biases for the first layer (input to hidden)
    key, subkey = jax.random.split(key)
    W1 = jax.random.normal(subkey, (input_size, hidden_size)) * 0.1
    b1 = jnp.zeros(hidden_size)

    # Weights and biases for the second layer (hidden to output)
    key, subkey = jax.random.split(key)
    W2 = jax.random.normal(subkey, (hidden_size, hidden_size)) * 0.1
    b2 = jnp.zeros(hidden_size)

    # Weights and biases for the second layer (hidden to output)
    key, subkey = jax.random.split(key)
    W3 = jax.random.normal(subkey, (hidden_size, output_size)) * 0.1
    b3 = jnp.zeros(output_size)

    return (W1, b1, W2, b2, W3, b3)


@jax.jit
def relu(x):
    """ReLU activation function"""
    return jnp.maximum(0, x)


@jax.jit
def forward(params, x):
    """Feedforward function. Returns the log probs for each action"""
    W1, b1, W2, b2, W3, b3 = params
    first = jnp.dot(x, W1) + b1
    hidden = relu(jnp.dot(first, W2) + b2)  # Hidden layer with ReLU
    output = jnp.dot(hidden, W3) + b3

    log_probs = jax.nn.log_softmax(output, axis=-1)  # get the log probs
    return log_probs
