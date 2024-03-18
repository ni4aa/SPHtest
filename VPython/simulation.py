import vpython as vp
import numpy as np
from tqdm import tqdm


def start_simulation(b):
    b.disabled = True
    b.text = 'In Process'
    for iter in tqdm(range(0, len(positions_of_all_time), 4)):
        for i in range(len(balls)):
            balls[i].pos = vp.vector(positions_of_all_time[iter, i, 0], positions_of_all_time[iter, i, 1],
                                     positions_of_all_time[iter, i, 2])
        vp.rate(1200)
        vp.sleep(0.001)

    b.disabled = False
    b.text = 'Repeat'


if __name__=="__main__":

    scene = vp.canvas(width=1300, height=700, background=vp.vector(1, 1, 1))
    scene.camera.pos = vp.vector(850, 150, 400)
    scene.camera.axis = vp.vector(-100, -100, -100)
    positions_of_all_time = np.load('damCollapse.npy', allow_pickle=True)
    balls = [vp.sphere(pos=vp.vector(ball[0], ball[1], ball[2]), radius=10, color=vp.color.blue)
             for ball in positions_of_all_time[0]]
    button = vp.button(bind=start_simulation, text='Start')
    while True:
        pass
