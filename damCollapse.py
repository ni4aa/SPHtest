import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm


PARTICLES = 400
DOMAIN_WIDTH = 1200
DOMAIN_HEIGHT = 300

DAM_BOUNDARY = 300
ITER_FOR_COLLAPSE = 700

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 15
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([[0, -0.1]])
N_OF_TRAMP = np.array([3, 1])
NORM_N = N_OF_TRAMP / np.linalg.norm(N_OF_TRAMP)

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2_500
ADD_PARTICLES_EVERY = 100

FIGURE_SIZE = (6, 3)
PLOT_EVERY = 4
SCATTER_DOT_SIZE = SMOOTHING_LENGTH * 10/2

DOMAIN_X_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_WIDTH - SMOOTHING_LENGTH])
DOMAIN_Y_LIM = np.array([SMOOTHING_LENGTH, DOMAIN_HEIGHT - SMOOTHING_LENGTH])

NORMALIZATION_DENSITY = (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH ** 9)
NORMALIZATION_PRESSURE_FORCE = -(45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH ** 6)
NORMALIZATION_VISCOUS_FORCE = (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH ** 6)


def set_positions(particles):
    positions = np.zeros((particles, 2))
    x_ptr = 0
    y_ptr = 0
    for i in range(particles):
        positions[i, 0] = 10 + (y_ptr % 2) * (SMOOTHING_LENGTH / 2) + x_ptr * SMOOTHING_LENGTH
        positions[i, 1] = 10 + y_ptr * SMOOTHING_LENGTH/1.1
        x_ptr += 1
        if positions[i, 0] + SMOOTHING_LENGTH >= DAM_BOUNDARY:
            y_ptr += 1
            x_ptr = 0

    return positions


def calculate_densities(neighbor_ids, distances):
    densities = np.zeros(len(neighbor_ids))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            influence = (SMOOTHING_LENGTH ** 2 - distances[i][j_in_list] ** 2) ** 3
            densities[i] += NORMALIZATION_DENSITY * influence

    return densities


def calculate_forces(neighbor_ids, distances, positions, pressures, velocities, densities):
    forces = np.zeros((len(neighbor_ids), 2))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            # Pressure force
            forces[i] += NORMALIZATION_PRESSURE_FORCE * (
                    -(positions[j] - positions[i]) / distances[i][j_in_list] *
                    (pressures[j] + pressures[i]) / (2 * densities[j]) *
                    (SMOOTHING_LENGTH - distances[i][j_in_list]) ** 2)

            # Viscous force
            forces[i] += NORMALIZATION_VISCOUS_FORCE * (
                    (velocities[j] - velocities[i]) / densities[j] * (SMOOTHING_LENGTH - distances[i][j_in_list]))
    # Force due to gravity
    forces += CONSTANT_FORCE

    return forces


def main():
    np.random.seed(0)
    positions = set_positions(300)
    # positions = 300 * np.random.random((PARTICLES, 2))
    velocities = np.zeros_like(positions)

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

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

        # Drop the element itself
        neighbor_ids = [np.delete(x, 0) for x in neighbor_ids]
        distances = [np.delete(x, 0) for x in distances]

        forces = calculate_forces(neighbor_ids, distances, positions, pressures, velocities, densities)

        # Euler Step
        velocities = velocities + TIME_STEP_LENGTH * forces / densities[:, np.newaxis]
        positions = positions + TIME_STEP_LENGTH * velocities

        # Enfore Boundary Conditions
        if iter <= ITER_FOR_COLLAPSE:
            out_of_right_boundary = positions[:, 0] > DAM_BOUNDARY

            velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
            positions[out_of_right_boundary, 0] = 300
        else:
            out_of_right_boundary = positions[:, 0] > DOMAIN_X_LIM[1]

            velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
            positions[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

        alpha = N_OF_TRAMP[1]/N_OF_TRAMP[0]
        tramp = ((positions[:, 0] < 650) & (positions[:, 0] > 550)) & \
                (positions[:, 1] < N_OF_TRAMP[1]/N_OF_TRAMP[0] * (positions[:, 0]-550)) & (velocities[:,0] > 0)

        if tramp.any():

            positions[tramp, 1] = (positions[tramp, 0] - 550) * alpha
            velocities[tramp, :] = np.array([ -DAMPING_COEFFICIENT *
                                              (velocity + (2 * NORM_N * (velocity @ NORM_N) / np.linalg.norm(velocity)))
                                              for velocity in velocities[tramp, :]])


        out_of_left_boundary = positions[:, 0] < DOMAIN_X_LIM[0]
        out_of_bottom_boundary = positions[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary = positions[:, 1] > DOMAIN_Y_LIM[1]

        velocities[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

        velocities[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

        if iter % PLOT_EVERY == 0:
            color = np.array([np.linalg.norm(velocity) for velocity in velocities])
            bar = plt.scatter(
                positions[:, 0],
                positions[:, 1],
                s=SCATTER_DOT_SIZE,
                c=color,
                cmap="cool",
            )
            if iter <= ITER_FOR_COLLAPSE:
                plt.plot([DAM_BOUNDARY, DAM_BOUNDARY], [0, DOMAIN_Y_LIM[1]], c='brown')
            else:
                plt.colorbar(bar)

            plt.plot([550, 650], [0, 100 * alpha], c='brown')
            plt.ylim(DOMAIN_Y_LIM)
            plt.xlim(DOMAIN_X_LIM)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


if __name__=="__main__":
    main()