import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as animation

TimedAnimation = animation.TimedAnimation
import time

def lerp(a,b,c):
    return a*c + b*(1.0-c)

def noise(scale = 1.0):
    return scale * np.random.randn()
#return 0.0

def control(cur, feed, dt):
    # currently, it simply interpolates with velocity
    # noise represents controller error

    #return feed/dt
    #return cur + 0.5 * feed/dt + noise(0.01) # TODO: scale noise?
    return lerp(cur, feed/dt, 0.2) * (1.0 + noise(0.01))

def measure(pos):
    # noise represents measurement error
    return pos * (1.0 + noise(0.1))

def arr(l):
    return np.asarray(l, dtype=np.float32)

def simulate(n):
    dims = 2 # dimensions 

    dt = 1. # time scale

    ts = [dt * _it for _it in range(n)]

    k_p = 0.1
    k_i = 0.4
    k_d = 0.01

    i = arr([0.0 for _ in range(dims)]) # integral accumulator
    d = arr([0.0 for _ in range(dims)]) # derivative term
    o = arr([0.0 for _ in range(dims)]) # objective 

    p_r = arr([10 * np.random.random() for _ in range(dims)]) # real y "position"
    p_m = p_r.copy() # measured y, copy p_r

    e_r = arr([o[_it] - p_r[_it] for _it in range(dims)]) # real error
    e_m = e_r.copy()

    v = arr([0. for _ in range(dims)]) # velocity

    # storing them

    p_rs = arr([p_r.copy() for _ in range(n)])
    p_ms = p_rs.copy()

    e_rs = arr([e_r.copy() for _ in range(n)])
    e_ms = e_rs.copy()

    vs = arr([list(v) for _ in range(n)])

    for t_idx, t in enumerate(ts): # repeat for duration
        for _it in range(dims): # 1-dimensional space
            #e_m = o - p_m
            #e_r = o - p_r
            #i += e_m * dt
            #d = e_m - e_ms[-1]
            #x = k_p * e_m + k_i * i/(t+dt) + k_d * d
            #v = control(v, x, dt)
            #p_r += v * dt + noise(0.1)
            #p_m = measure(p_r)

            e_m[_it] = o[_it] - p_m[_it] # measured error
            e_r[_it] = o[_it] - p_r[_it] # real error

            i[_it] += e_m[_it] * dt # integral term, normalized
            d[_it] = (e_m[_it] - e_ms[t_idx][_it])/dt # derivative term
            x = k_p * e_m[_it] + k_i * (i[_it]/(t+dt)) + k_d * d[_it]

            v[_it] = control(v[_it], x, dt)
            # noise represents environmental noise
            p_r[_it] += v[_it] * dt + noise(0.1) # TODO : scale noise?
            p_m[_it] = measure(p_r[_it])

        #np.concatenate_rs.append(list(e_r))

        #currently editing here
        p_rs[t_idx] = p_r
        p_ms[t_idx] = p_m

        vs[t_idx] = v
    return p_rs, p_ms

    #plt.plot(ts,e_rs[1:,0] / e_rs[0,0],label='e_rs')
    #plt.plot(ts,e_ms[1:,0] / e_rs[0,0],label='e_ms')
    #plt.plot(ts,vs[1:,0] / e_rs[0,0],label='velocity')


class CometAnimation(TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        self.l = 500 
        p_rs,p_ms = simulate(self.l)

        ax.set_xlim(np.min(p_rs[:,0]), np.max(p_rs[:,0]))
        ax.set_ylim(np.min(p_rs[:,1]), np.max(p_rs[:,1]))

        self.p_rs = p_rs
        self.p_ms = p_ms

        self.comet1, = plt.plot([],[],label='real')
        self.comet2, = plt.plot([],[],label='measured')
        self.centers, = plt.plot([],[],'or',label='centers')
        self.origin, = plt.plot([0],[0],"*b")

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        self.comet1.set_data(self.p_rs[1:i, 0], self.p_rs[1:i, 1])
        self.comet2.set_data(self.p_ms[1:i, 0], self.p_ms[1:i, 1])
        self.centers.set_data(
                [self.p_rs[i,0],self.p_ms[i,0]],
                [self.p_rs[i,1],self.p_ms[i,1]],
                )

        self._drawn_artists = [self.comet1, self.comet2, self.centers, self.origin]

    def new_frame_seq(self):
        return iter(range(self.l))

    def _init_draw(self):
        plts = [self.comet1, self.comet2, self.centers]
        for p in plts:
            p.set_data([], [])

def main():
    ani = CometAnimation()
    ani.save('pid.mp4')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
