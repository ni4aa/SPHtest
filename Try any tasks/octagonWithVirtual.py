from functions import *
from Walls import Wall
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm


DOMAIN_WIDTH = 300
DOMAIN_HEIGHT = 300
SMOOTHING_LENGTH = 40
PARTICLES = 100

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2500
FIGURE_SIZE = (4, 4)
PLOT_EVERY = 4
SCATTER_DOT_SIZE = 200

if __name__ == '__main__':
    array_of_walls = np.empty(8, dtype=Wall)
    angle = np.deg2rad(90)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    for i in range(8):
        if i == 0:
            array_of_walls[0] = Wall([0, 1], [0, 0], [300, 0])
        else:
            norm = array_of_walls[i - 1].get_norm()
            new_norm = rotation_matrix @ norm
            start, end = array_of_walls[i - 1].get_pos()
            new_start = end
            new_end = new_start + new_norm[::-1] * 300
            array_of_walls[i] = Wall(new_norm, new_start, new_end)

    boundary = np.zeros((80, 2))
    number_of_boundary = 0
    for i in range(0, len(boundary), 10):
        start, end = array_of_walls[number_of_boundary].get_pos()
        t = np.linspace(0, 1, 10)[:, None]
        segment_points = start + t * (end - start)
        boundary[i:i+10] = np.array(segment_points)
        number_of_boundary += 1

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    np.random.seed(12)
    positions = np.array([[0, 150]]) + 300 * np.random.random((PARTICLES, 2))

    velocities = np.zeros_like(positions)

    for iter in tqdm(range(N_TIME_STEPS)):

        positions_with_boundaries = np.concatenate((positions, boundary), axis=0)

        velocities_with_boundaries = np.concatenate((positions, np.zeros((len(boundary),2))), axis=0)

        neighbor_ids, distances = neighbors.KDTree(
            positions_with_boundaries,
        ).query_radius(
            positions_with_boundaries,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True,
        )

        densities = calculate_densities(neighbor_ids, distances)

        pressures = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)

        neighbor_ids = [np.delete(x, 0) for x in neighbor_ids]
        distances = [np.delete(x, 0) for x in distances]

        accelerations = calculate_acceleration(neighbor_ids, distances, positions_with_boundaries, pressures,
                                               velocities_with_boundaries, densities)

        # Euler Step
        velocities = velocities + TIME_STEP_LENGTH * accelerations[:len(velocities)]
        positions = positions + TIME_STEP_LENGTH * velocities

        if iter % PLOT_EVERY == 0:
            color = np.array([np.linalg.norm(velocity) for velocity in velocities])
            for i in range(8):
                array_of_walls[i].plot_wall()
            plt.scatter(
                positions[:, 0],
                positions[:, 1],
                s=SCATTER_DOT_SIZE,
                c=color,
                cmap="cool",
            )
            plt.xlim(-200, 500)
            plt.ylim(0, 700)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()



