import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from scipy.spatial import cKDTree
from tqdm import tqdm


DOMAIN_WIDTH = 300
DOMAIN_HEIGHT = 300
SMOOTHING_LENGTH = 40
PARTICLES = 100

PARTICLE_MASS = 1
BASE_DENSITY = 1
ISOTROPIC_EXPONENT = 10  # нужно что-то другое
DYNAMIC_VISCOSITY = 0.2
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([0, 0])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2500
FIGURE_SIZE = (4, 4)
PLOT_EVERY = 4
SCATTER_DOT_SIZE = 600

DOMAIN_X_LIM = np.array([0, DOMAIN_WIDTH])
DOMAIN_Y_LIM = np.array([0, DOMAIN_HEIGHT])

NORMALIZATION_KERNEL = 315 / (64 * np.pi * SMOOTHING_LENGTH ** 9)

def set_positions():
    positions = np.zeros((PARTICLES, 2))
    x_ptr = 0
    y_ptr = 0
    for i in range(PARTICLES):
        positions[i, 0] = DOMAIN_WIDTH/20 * x_ptr + DOMAIN_WIDTH/5 * 2
        positions[i, 1] = DOMAIN_HEIGHT/20 * y_ptr + DOMAIN_HEIGHT/5 * 2
        x_ptr += 1
        if (i+1) % 10 == 0:
            y_ptr += 1
            x_ptr = 0

    return positions


def calculate_densities(neighbor_ids, distances):
    densities = np.zeros(len(neighbor_ids))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            influence = NORMALIZATION_KERNEL * (SMOOTHING_LENGTH ** 2 - distances[i][j_in_list] ** 2) ** 3
            densities[i] += PARTICLE_MASS * influence

    return densities


def calculate_acceleration(neighbor_ids, distances, positions, pressures, velocities, densities):
    accelerations = np.zeros((len(neighbor_ids), 2))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            influence = NORMALIZATION_KERNEL * 6 * \
                        (SMOOTHING_LENGTH ** 2 - distances[i][j_in_list] ** 2) ** 2
            # pressure forces
            accelerations[i] += - PARTICLE_MASS * (pressures[i] + pressures[j])/(densities[i] * densities[j]) * influence * \
                         (positions[i]-positions[j])

            accelerations[i] += DYNAMIC_VISCOSITY * PARTICLE_MASS * (velocities[i] - velocities[j]) * pressures[i] /\
                         (densities[i]*densities[j]) * influence

        accelerations[i] += CONSTANT_FORCE / densities[i, np.newaxis]

    return accelerations


if __name__ == '__main__':
    # positions = set_positions()

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    np.random.seed(12)
    positions = 300 * np.random.random((PARTICLES, 2))

    velocities = np.zeros_like(positions)

    for iter in tqdm(range(N_TIME_STEPS)):

        neighbor_ids, distances = neighbors.KDTree(
            positions,
        ).query_radius(
            positions,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True,
        )

        densities = calculate_densities(neighbor_ids, distances)

        pressures = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)

        neighbor_ids = [np.delete(x, 0) for x in neighbor_ids]
        distances = [np.delete(x, 0) for x in distances]

        accelerations = calculate_acceleration(neighbor_ids, distances, positions, pressures, velocities, densities)

        # Euler Step
        velocities = velocities + TIME_STEP_LENGTH * accelerations
        positions = positions + TIME_STEP_LENGTH * velocities

        # Enforce Boundary Conditions
        out_of_left_boundary = positions[:, 0] < DOMAIN_X_LIM[0]
        out_of_right_boundary = positions[:, 0] > DOMAIN_X_LIM[1]
        out_of_bottom_boundary = positions[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary = positions[:, 1] > DOMAIN_Y_LIM[1]

        velocities[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

        velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

        velocities[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

        if iter % PLOT_EVERY == 0:
            color = np.array([np.linalg.norm(velocity) for velocity in velocities])
            plt.scatter(
                positions[:, 0],
                positions[:, 1],
                s=SCATTER_DOT_SIZE,
                c=color,
                cmap="cool",
            )
            plt.ylim(DOMAIN_Y_LIM)
            plt.xlim(DOMAIN_X_LIM)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()

