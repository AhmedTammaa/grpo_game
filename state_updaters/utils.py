import jax
from jax.numpy import jnp
from constants import K_spring, mu_friction, g


@jax.jit
def calculate_forces(info):
    """Given the current state, calculates the forces on the three vertices due to
    1. Spring forces
    2. Friction
    Returns the force on each vertex in Shape (batch_size,3,2)"""

    # Input information:
    positions, velocities, masses, pumps, muscles = info
    # Shapes of things:
    # Vertex Indexed Things:
    #   positions (batch_size, 3, 2)
    #   velocities (batch_size, 3, 2)
    #   masses (batch_size, 3)
    #   pumps (batch_size, 3)
    #     [0,1] valued where 0 represents max shrink, 1 represent max grow
    # Edge indexed things: Edge Order 01, 02, 12
    #   muscles (batch_size, 3)
    #     [0,1] valued  where 0 represents minimum lenght, 1 represents max length

    ###--EDGE STUFF--

    # Compute edge vectors for all triangles in the batch
    edge_01 = (
        positions[..., 1, :] - positions[..., 0, :]
    )  # Vector from point 0 to point 1
    edge_02 = (
        positions[..., 2, :] - positions[..., 0, :]
    )  # Vector from point 0 to point 2
    edge_12 = (
        positions[..., 2, :] - positions[..., 1, :]
    )  # Vector from point 1 to point 2

    # Compute the current lengths of the edges
    length_01 = jnp.linalg.norm(edge_01, axis=-1)  # Shape: (batch_size,)
    length_02 = jnp.linalg.norm(edge_02, axis=-1)  # Shape: (batch_size,)
    length_12 = jnp.linalg.norm(edge_12, axis=-1)  # Shape: (batch_size,)

    # Normalize edge vectors to get unit directions.
    # make sure lengths are at least epsilon to aviod diveisoon by zero
    epsilon = 1e-8
    dir_01 = edge_01 / jnp.maximum(
        length_01[..., None], epsilon
    )  # Shape: (batch_size, 2)
    dir_02 = edge_02 / jnp.maximum(
        length_02[..., None], epsilon
    )  # Shape: (batch_size, 2)
    dir_12 = edge_12 / jnp.maximum(
        length_12[..., None], epsilon
    )  # Shape: (batch_size, 2)

    # Compute spring forces based on: F = -K * (L - L_eq) * direction

    edge_lengths_eq = L_min + muscles * (
        L_max - L_min
    )  # equilbirum lengths based on muscle status

    s_force_01 = (
        -K_spring * (length_01 - edge_lengths_eq[..., 0])[..., None] * dir_01
    )  # Shape: (batch_size, 2)
    s_force_02 = (
        -K_spring * (length_02 - edge_lengths_eq[..., 1])[..., None] * dir_02
    )  # Shape: (batch_size, 2)
    s_force_12 = (
        -K_spring * (length_12 - edge_lengths_eq[..., 2])[..., None] * dir_12
    )  # Shape: (batch_size, 2)

    # Initialize forces tensor for all vertices
    # forces is indexed by the vertices now.
    forces = jnp.zeros_like(positions)  # Shape: (batch_size, 3, 2)

    # Accumulate spring forces for each vertex
    # Note that you have to use x = x.at[ ix ].add[ delta_x ] rather than x += delta_x because jax arrays are imutable

    forces = forces.at[..., 0, :].add(
        -s_force_01 - s_force_02
    )  # Point 0 affected by edges 01 and 02
    forces = forces.at[..., 1, :].add(
        s_force_01 - s_force_12
    )  # Point 1 affected by edges 01 and 12
    forces = forces.at[..., 2, :].add(
        s_force_02 + s_force_12
    )  # Point 2 affected by edges 02 and 12

    ###--FRICTION FORCES--
    # Compute the magnitudes of the velocities for each vertex in the batch
    speed = jnp.linalg.norm(velocities, axis=-1)  # Shape: (batch_size, 3)
    epsilon = 1e-8
    velocity_directions = velocities / jnp.maximum(speed[..., None], epsilon)
    # Safe divide by numbers close to zero. Shape: (batch_size, 3, 2)

    # Compute the kinetic friction force magnitudes
    kinetic_friction_magnitudes = (
        mu_friction * masses * g
    )  # Shape: (batch_size, 3)

    # Add friction forces opposite the velocity direction to each vertex
    forces = (
        forces - kinetic_friction_magnitudes[..., None] * velocity_directions
    )
    return forces
