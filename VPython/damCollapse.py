import vpython as vp
import numpy as np
from sklearn import neighbors
from tqdm import tqdm


DOMAIN_WIDTH = 1200
DOMAIN_HEIGHT = 300
DOMAIN_DEEP = 150
SMOOTHING_LENGTH = 20
PARTICLES = 1000

PARTICLE_MASS = 1
BASE_DENSITY = 1
ISOTROPIC_EXPONENT = 12
DYNAMIC_VISCOSITY = 2
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([0, -0.04, 0])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 1000
FIGURE_SIZE = (4, 4)
PLOT_EVERY = 4
SCATTER_DOT_SIZE = 600


DOMAIN_X_LIM = np.array([-150/2, DOMAIN_WIDTH/2])
DOMAIN_Y_LIM = np.array([-DOMAIN_HEIGHT/2, DOMAIN_HEIGHT/2])
DOMAIN_Z_LIM = np.array([-DOMAIN_DEEP/2, DOMAIN_DEEP/2])


NORMALIZATION_KERNEL = 315 / (64 * np.pi * SMOOTHING_LENGTH ** 9)


def set_positions(width, height, deep):
    x_coords = np.linspace(DOMAIN_X_LIM[0], DOMAIN_X_LIM[1], width)
    y_coords = np.linspace(DOMAIN_Y_LIM[0], DOMAIN_Y_LIM[1]/2, height)
    z_coords = np.linspace(DOMAIN_Z_LIM[0], DOMAIN_Z_LIM[1], deep)

    # Создаем сетку координат точек
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)

    # Создаем массив координат точек в виде (x, y, 0)
    point_cloud = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    return point_cloud


def calculate_densities(neighbor_ids, distances):
    densities = np.zeros(len(neighbor_ids))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            influence = NORMALIZATION_KERNEL * (SMOOTHING_LENGTH ** 2 - distances[i][j_in_list] ** 2) ** 3
            densities[i] += PARTICLE_MASS * influence

    return densities


def calculate_acceleration(neighbor_ids, distances, positions, pressures, velocities, densities):
    accelerations = np.zeros((len(neighbor_ids), 3))

    for i in range(len(neighbor_ids)):
        for j_in_list, j in enumerate(neighbor_ids[i]):
            influence = NORMALIZATION_KERNEL * 6 * \
                        (SMOOTHING_LENGTH ** 2 - distances[i][j_in_list] ** 2) ** 2
            # pressure forces
            accelerations[i] += - PARTICLE_MASS * (pressures[i] + pressures[j])/(densities[i] * densities[j]) * influence * \
                         (positions[i]-positions[j])

            accelerations[i] += DYNAMIC_VISCOSITY * PARTICLE_MASS * (velocities[i] - velocities[j]) / distances[i][
                j_in_list] * pressures[i] / \
                                (densities[i] * densities[j]) * influence

        accelerations[i] += CONSTANT_FORCE / densities[i, np.newaxis]

    return accelerations


def start_simulation(b):
    b.disabled = True
    b.text = 'In Process'
    for iter in tqdm(range(N_TIME_STEPS)):
        for i in range(len(balls)):
            balls[i].pos = vp.vector(positions_of_all_time[iter, i, 0], positions_of_all_time[iter, i, 1],
                                     positions_of_all_time[iter, i, 2])
        vp.rate(100)
        vp.sleep(0.01)

    b.disabled = False
    b.text = 'Repeat'


if __name__ == '__main__':
    scene = vp.canvas(width=700, height=700, background=vp.vector(1,1,1))

    positions = np.load('positions.npy')
    positions_of_all_time = np.empty((N_TIME_STEPS, len(positions), 3), dtype=np.ndarray)

    balls = [vp.sphere(pos=vp.vector(ball[0], ball[1], ball[2]), radius=10, color=vp.color.blue)
             for ball in positions]
    velocities = np.zeros_like(positions)

    for iter in tqdm(range(N_TIME_STEPS)):

        positions_of_all_time[iter, :] = positions[:]

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
        out_of_start_boundary = positions[:, 2] < DOMAIN_Z_LIM[0]
        out_of_end_boundary = positions[:, 2] > DOMAIN_Z_LIM[1]

        velocities[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

        velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

        velocities[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

        velocities[out_of_start_boundary, 2] *= DAMPING_COEFFICIENT
        positions[out_of_start_boundary, 2] = DOMAIN_Z_LIM[0]

        velocities[out_of_end_boundary, 2] *= DAMPING_COEFFICIENT
        positions[out_of_end_boundary, 2] = DOMAIN_Z_LIM[1]

    np.save("damCollapse", positions_of_all_time)
    button = vp.button(bind=start_simulation, text='Start')

    while True:
        pass

