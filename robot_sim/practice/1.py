import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np

base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
def surface_callback(x,y):
    z=2*x+1*y
    return z
rng=[[.06,.07],[.08,.09]]
gm.gen_surface(surface_callback, rng, granularity=.01).attach_to(base)
base.run()









