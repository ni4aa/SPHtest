import numpy as np
import numpy.typing as np_type
import matplotlib.pyplot as plt


class Wall:
    def __init__(self, norm: np_type.NDArray(2), start: np_type.NDArray(2), end: np_type.NDArray(2)):
        self.norm = np.array(norm) / np.linalg.norm(norm)
        self.refl_matrix = np.array([[1 - 2 * self.norm[0]**2, -2 * self.norm[0] * self.norm[1]],
                                     [-2 * self.norm[0] * self.norm[1], 1 - self.norm[1]**2]])
        self.start = np.array(start)
        self.end = np.array(end)
        t = np.linspace(0, 1, 10)[:, None]
        self.points = np.array(self.start + t * (self.end - self.start))


    def get_norm(self):
        return self.norm

    def get_pos(self):
        return self.start, self.end

    def plot_wall(self):
        plt.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]])

if __name__=="__main__":
    array_of_walls = np.empty(8, dtype=Wall)
    angle = np.deg2rad(45)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    for i in range(8):
        if i == 0:
            array_of_walls[0] = Wall([0, 1], [0, 0], [1, 0])
            array_of_walls[i].plot_wall()
            plt.draw()
        else:
            norm = array_of_walls[i-1].get_norm()
            new_norm = rotation_matrix @ norm
            start, end = array_of_walls[i-1].get_pos()
            new_start = end
            new_end = new_start + new_norm[::-1]

            array_of_walls[i] = Wall(new_norm, new_start, new_end)
            array_of_walls[i].plot_wall()
            plt.draw()

    plt.show()