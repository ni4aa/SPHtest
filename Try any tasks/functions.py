import numpy as np

SMOOTHING_LENGTH = 40
PARTICLE_MASS = 1
BASE_DENSITY = 1
ISOTROPIC_EXPONENT = 10  # нужно что-то другое
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([0, 0])

NORMALIZATION_KERNEL = 315 / (64 * np.pi * SMOOTHING_LENGTH ** 9)

from Consts import *


def distance_to_segment(point, segment_start, segment_end):
    l2 = np.sum((segment_start - segment_end)**2)
    if l2 == 0:
        return np.sqrt(np.sum((point - segment_start)**2))
    t = np.max(0, np.min(1, np.dot(point - segment_start, segment_end - segment_start) / l2))
    projection = segment_start + t * (segment_end - segment_start)
    return np.sqrt(np.sum((point - projection)**2))



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

            accelerations[i] += DYNAMIC_VISCOSITY * PARTICLE_MASS * (velocities[i] - velocities[j]) / distances[i][
                j_in_list] * pressures[i] / \
                                (densities[i] * densities[j]) * influence


        accelerations[i] += CONSTANT_FORCE / densities[i, np.newaxis]

    return accelerations
