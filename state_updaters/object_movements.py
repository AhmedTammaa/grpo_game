import jax
import jax.numpy as jnp

from constants import M_min, M_total, K_mass, dt
from state_updaters import calculate_forces


@jax.jit
def triangle_update(iter_num, info):
    """Applies one time step, of time dt, to the positions, velocities, masses of the triangle creature
    The input is Setup so that it can work with jax.lax for loops for efficnency. iter_num is never used
    """

    # Input information:
    positions, velocities, masses, pumps, muscles = info
    # Shapes:
    # Vertex Indexed Things:
    #   positions (batch_size, 3, 2)
    #   velocities (batch_size, 3, 2)
    #   masses (batch_size, 3)
    #   pumps (batch_size, 3)
    #     [0,1] valued where 0 represents max shrink, 1 represent max grow
    # Edge indexed things: Edge Order 01, 02, 12
    #   muscles (batch_size, 3)
    #     [0,1] valued  where 0 represents minimum lenght, 1 represents max length

    # Second-Order Runge-Kutta Method (RK2 - Midpoint Method)
    # -------------------------------------------------------
    # 1. Compute acceleration at the initial position and velocity.
    # 2. Estimate velocity and position at the midpoint using half a time step.
    # 3. Compute acceleration at this midpoint.
    # 4. Use the midpoint acceleration to update velocity and position for the full step.
    # - Requires 2 acceleration function calls per step.
    # - Global error is O(dt^2), making it much more accurate than Forward Euler (O(dt)).

    # Convert "pumps" to fractions, e.g. 0,1,1 -> 0,1/2,1/2. epsilon is there so that (0,0,0) -> (1/3,1/3,1/3)
    # this has the property that all entries are in [0,1] and they sum to 1.
    epsilon = 1e-6
    pumps_fraction = (pumps + epsilon) / jnp.sum(
        pumps + epsilon, axis=-1, keepdims=True
    )

    # Compute equilbrium mass amounts by distributing available weight by pump fractions
    masses_eq = M_min + pumps_fraction * (M_total - 3.0 * M_min)

    # RK2 method starts here

    # --Initial Calculations
    forces = calculate_forces((positions, velocities, masses, pumps, muscles))
    accelerations = forces / masses[..., None]  # Shape: (3, 2)

    # --Halfway point calculations
    masses_half = masses + (masses - masses_eq) * jnp.expm1(-K_mass * 0.5 * dt)
    # note expm1 is the function exp(x)-1.
    # This formula is the exact solution do dm/dt = K(m - m_eq) at time 0.5*dt

    positions_half = (
        positions + 0.5 * dt * velocities
    )  # halfway position estimate
    velocities_half = (
        velocities + 0.5 * dt * accelerations
    )  # halfway velocity estimate

    forces_half = calculate_forces(
        (positions_half, velocities_half, masses_half, pumps, muscles)
    )
    accelerations_half = forces_half / masses_half[..., None]  # Shape: (3, 2)

    # --Final calcualtions by using the midpoint
    masses_final = masses + (masses - masses_eq) * jnp.expm1(
        -K_mass * dt
    )  # exact solution
    positions_final = (
        positions + dt * velocities_half
    )  # estimate using midpoint velocity
    velocities_final = (
        velocities + dt * accelerations_half
    )  # estimate using midpoint accelerations

    # Note pumps, muscles are unchanged, but leave them in here for compatability in jax.lax for-loops
    return positions_final, velocities_final, masses_final, pumps, muscles
