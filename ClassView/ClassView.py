import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm


"""
Пока что это все грубо говоря, только наброски, надо еще много додумывать
"""


class Particle:
    def __init__(self, pos, vel, mass, pressure):
        self.position = pos
        self.velocity = vel
        self.mass = mass
        self.density = 0
        self.pressure = pressure

class Wall:
    def __init__(self, start, end, norm, smoothing_length):
        self.start = np.array(start)
        self.end = np.array(end)
        self.norm = np.array(norm)
        self.norm = self.norm / np.linalg.norm(self.norm)
        self.len = np.linalg.norm(self.end-self.start)
        t = np.arange(0, self.len, smoothing_length)
        self.particles = np.array([self.start + t[i] * (self.end-self.start)/self.len for i in range(len(t))])

    def get_pos(self):
        return self.start, self.end

    def plot_wall(self, plot_particle=False):
        plt.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]])
        if plot_particle:
            for particle in self.particles:
                plt.scatter(*particle, 10)


class ParticleSystem:
    def __init__(self, array_of_particles, array_of_walls, dynamic_viscosity, isotropic_exponent, base_pressure,
                 base_density, const_force, time_step, n_steps, scatter_dot_size, smoothing_length, length_for_wall):
        self.array_of_particles = array_of_particles
        self.array_of_walls = array_of_walls
        self.boundaries = np.array([wall.particles for wall in array_of_walls])

        self.dynamic_viscosity = dynamic_viscosity
        self.isotropic_exponent = isotropic_exponent
        self.base_pressure = base_pressure
        self.base_density = base_density
        self.const_force = np.array(const_force)
        self.length_for_wall = length_for_wall
        self.D = 0

        self.time_step = time_step
        self.n_steps = n_steps
        self.scatter_dot_size = scatter_dot_size

        self.smoothing_length = smoothing_length
        self.NORMALIZATION_KERNEL = 315 / (64 * np.pi * self.smoothing_length ** 9)

    def __get_new_pos(self):
        self.positions = np.array([particle.position for particle in self.array_of_particles])
        self.neighbor_ids, self.distances = neighbors.KDTree(
            self.positions,
        ).query_radius(
            self.positions,
            self.smoothing_length,
            return_distance=True,
            sort_results=True,
        )

    def __calculate_density(self):
        for i in range(len(self.neighbor_ids)):
            self.array_of_particles[i].density = 0
            for j_in_list, j in enumerate(self.neighbor_ids[i]):
                influence = self.NORMALIZATION_KERNEL * (self.smoothing_length ** 2 - self.distances[i][j_in_list] ** 2) ** 3
                self.array_of_particles[i].density += self.array_of_particles[j].mass * influence

        self.neighbor_ids = [np.delete(x, 0) for x in self.neighbor_ids]
        self.distances = [np.delete(x, 0) for x in self.distances]

    def __calculate_acceleration(self):
        for i in range(len(self.neighbor_ids)):
            self.array_of_particles[i].pressure = self.base_pressure + self.isotropic_exponent * \
                                                  (self.array_of_particles[i].density - self.base_density)

        self.accelerations = np.zeros((len(self.array_of_particles), 2))

        for i in range(len(self.array_of_particles)):
            for j_in_list, j in enumerate(self.neighbor_ids[i]):
                influence = self.NORMALIZATION_KERNEL * 6 * \
                            (self.smoothing_length ** 2 - self.distances[i][j_in_list] ** 2) ** 2

                self.accelerations[i] += - self.array_of_particles[j].mass * \
                                    (self.array_of_particles[i].pressure + self.array_of_particles[j].pressure) / \
                                    (self.array_of_particles[i].density * self.array_of_particles[j].density) * \
                                    influence * (self.positions[i] - self.positions[j])

                self.accelerations[i] += self.dynamic_viscosity * self.array_of_particles[i].mass * \
                                    (self.array_of_particles[i].velocity - self.array_of_particles[i].velocity)/\
                                    self.distances[i][j_in_list] * \
                                    self.array_of_particles[i].pressure / \
                                    (self.array_of_particles[i].density * self.array_of_particles[j].density) * influence

            self.accelerations[i] += self.const_force / self.array_of_particles[i].density

    def __calculate_wall_acceleration(self):
        for boundary in self.boundaries:
            ids, distances_to_wall = neighbors.KDTree(
                boundary,
            ).query_radius(
                self.positions,
                self.length_for_wall,
                return_distance=True,
                sort_results=True,
            )

            for i in range(len(self.array_of_particles)):
                for j_in_list, j in enumerate(ids[i]):
                    self.accelerations[i] += self.D * ((self.length_for_wall/distances_to_wall[i][j_in_list])**12 -
                                                       (self.length_for_wall/distances_to_wall[i][j_in_list])**4) / \
                                             (self.array_of_particles[i].density * distances_to_wall[i][j_in_list]**2) * \
                                             (self.positions[i] - boundary[j])

    def __calculate_next_step(self):
        max_vel = 0
        for i in range(len(self.array_of_particles)):
            self.array_of_particles[i].velocity += self.time_step * self.accelerations[i]
            max_vel = max(max_vel, self.array_of_particles[i].velocity @ self.array_of_particles[i].velocity)
            self.array_of_particles[i].position += self.time_step * self.array_of_particles[i].velocity

        self.D = max_vel

    def __plot_system(self):
        for i in range(len(self.array_of_walls)):
            self.array_of_walls[i].plot_wall()

        color = np.sqrt(np.linspace(0, self.D, 100))
        plt.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            s=self.scatter_dot_size,
            c=color,
            cmap="cool",
        )
        plt.xlim(0, 300)
        plt.ylim(0, 300)

        plt.xticks([], [])
        plt.yticks([], [])
        plt.tight_layout()
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def start_simulation(self):
        for iter in tqdm(range(self.n_steps)):
            self.__get_new_pos()
            if iter % 4 == 0:
                self.__plot_system()

            self.__calculate_density()
            self.__calculate_acceleration()
            self.__calculate_wall_acceleration()
            self.__calculate_next_step()


if __name__ == '__main__':
    smoothing_length = 40
    array_of_walls = np.empty(4, dtype=Wall)
    array_of_walls[0] = Wall([0, 0], [300, 0], [0, 1], smoothing_length)
    array_of_walls[1] = Wall([300, 0], [300, 300], [-1, 0], smoothing_length)
    array_of_walls[2] = Wall([300, 300], [0, 300], [0, -1], smoothing_length)
    array_of_walls[3] = Wall([0, 300], [0, 0], [1, 0], smoothing_length)

    size = 100

    np.random.seed(12)
    positions = np.array([[250, 250]]) * np.random.random((size, 2))
    mass = np.ones(len(positions))
    vel = np.zeros_like(positions)
    pressure = np.ones(len(positions))

    array_of_particles = np.empty(100, dtype=Particle)
    for i in range(len(positions)):
        array_of_particles[i] = Particle(positions[i], vel[i], mass[i], pressure[i])

    FirstSystem = ParticleSystem(
        array_of_particles, array_of_walls,
        dynamic_viscosity=0.2,
        isotropic_exponent=10,
        base_pressure=0,
        base_density=1,
        const_force=np.array([0, 0]),
        time_step=0.01,
        n_steps=2_500,
        scatter_dot_size=300,
        smoothing_length=smoothing_length,
        length_for_wall=smoothing_length/3
    )
    FirstSystem.start_simulation()

