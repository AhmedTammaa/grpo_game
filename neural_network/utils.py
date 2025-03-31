import jax
import jax.numpy as jnp


@jax.jit
def normalize_points(points):
    # given the vertices of a triangle return their relative positions by subtracting off the centroid
    centroid = jnp.mean(
        points, axis=-2, keepdims=True
    )  # Shape: (batch_size, 1, 2)
    normalized_points = points - centroid
    return normalized_points


@jax.jit
def get_input(my_points, my_velocities, my_masses, my_pumps, my_muscles):
    # This is the input to the neural network it consits of 15 numbers:
    # Next 3 entries: Relative x-coorindates of poitns (so it can know which of its feet are left/right)
    # Next 3 entries: Rleative y-coordinattes of points (so it can know which of its feet are up/down)
    # Next 3 entries: previous pumps setting as 0 or 1
    # Next 3 entries: previous muscle setting as 0 or 1
    # Last bonus entry: y-coordinates of the centroid points (so it can know how far up/down it is)

    y_centroid = jnp.mean(my_points[..., 1], axis=-1)

    # expand_dims on the first one is needed on the y_centroid to prevent it from working when its a scalar
    result = jnp.concatenate(
        [
            normalize_points(my_points)[..., 0],  # Ensure at least 1D
            normalize_points(my_points)[..., 1],  # Ensure at least 1D
            my_pumps,
            my_muscles,
            jnp.expand_dims(y_centroid, axis=-1),
        ],
        axis=-1,
    )
    return result  # jnp.concatenate([y_centroid,normalize_points(my_points)[...,0],normalize_points(my_points)[...,1],my_pumps, my_muscles],axis=-1)


@jax.jit
def encode_pumps_muscles(pumps, muscles):
    """Encode pumps and muscles as a one-hot vector"""
    # Flatten the 3-bit values for pumps and muscles into a single 6-bit vector
    # go from shape (batch_size, 3) to (batch_size)
    pumps_flat = (pumps[..., 0] << 2) + (pumps[..., 1] << 1) + pumps[..., 2]
    muscles_flat = (
        (muscles[..., 0] << 2) + (muscles[..., 1] << 1) + muscles[..., 2]
    )
    # We will use the combination of pumps and muscles as a label (from 0 to 63)
    return jnp.array(
        [pumps_flat * 8 + muscles_flat]
    )  # 64 possible configurations


@jax.jit
def decode_pumps_muscles(encoded_value):
    """Decode an integer (0 to 63) into pump and muscle settings."""

    # Extract the pumps (first 3 bits)
    pumps_flat = encoded_value // 8  # Integer division by 8
    pumps = jnp.stack(
        [(pumps_flat >> i) & 1 for i in range(2, -1, -1)], axis=-1
    )

    # Extract the muscles (last 3 bits)
    muscles_flat = encoded_value % 8  # Get the last 3 bits using modulus 8
    muscles = jnp.stack(
        [(muscles_flat >> i) & 1 for i in range(2, -1, -1)], axis=-1
    )

    return pumps, muscles
