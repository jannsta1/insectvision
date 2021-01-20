from mayavi import mlab
import numpy as np

res = 250. #desired resolution (number of samples on sphere)
phi,theta = np.mgrid[0:np.pi:np.pi/res, 0:np.pi:np.pi/res]

x=np.cos(theta) * np.sin(phi)
y=np.sin(theta) * np.sin(phi)
z=np.cos(phi)

# mlab.mesh(x,y,z,color=(1,1,1))

from mayavi.mlab import *

def test_contour_surf():
    """Test contour_surf on regularly spaced co-ordinates like MayaVi."""
    def f(x, y):
        sin, cos = np.sin, np.cos
        return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
    s = contour_surf(x, y, f)
    return s

s = test_contour_surf()