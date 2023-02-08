import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
os.popen("rm *pdf *png")
from math import pi
from math import sin
from math import cos

 # https://scipython.com/blog/visualizing-a-vector-field-with-matplotlib/
mu=1

def E(q, r0, x, y):
    """Return the electric field vector E=(Ex,Ey) due to charge q at r0."""
    # den=r^3 to charge
    den = np.hypot(x-r0[0], y-r0[1])**3
    ret= q * (x - r0[0]) / den, q * (y - r0[1]) / den
    return ret


# for moment in [3]:
for moment in [1,2,3]:

    # Grid of x, y points
    n = 200
    nx, ny = n,n
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Create a multipole with nq charges of alternating sign, equally spaced
    # on the unit circle.
    nq = 2**moment
    if moment==3: nq=6
    charges = []
    for i in range(nq):
        q = i%2 * 2 - 1
        angle = 2*pi*i/nq 
        r=1
        if moment==1:
            angle+= pi/2
            r=0.9
        if moment==2:
            angle+= pi/4
            r=1.1
        if moment==3:
            angle+= pi/2
            r=0.9
            # r=1.25
        charges.append((q, (r*np.cos(angle), r*np.sin(angle))))

    # Electric field vector, E=(Ex, Ey), as separate components
    Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
    for charge in charges:
        ex, ey = E(*charge, x=X, y=Y)
        Ex += ex
        Ey += ey

    fig = plt.figure(figsize=(5,5), dpi=100)
    ax = plt.gca()

    # Plot the streamlines with an appropriate colormap and arrow style
    sp=[]
    if moment==1:
        t = np.array([[-1,0],[1,0]])
        sp+=list(t*0.0)
        sp+=list(t*0.2)
        sp+=list(t*0.4)
        sp+=list(t*0.6)
        sp+=list(t*0.8)
        sp+=list(t*1.0)
        sp+=list(t*1.2)
        sp+=list(t*1.4)
        sp+=list(t*1.6)
        sp+=list(t*1.8)
        sp+=list(t*2.0)
    if moment==2:
        t = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        sp+=list(t*0.2)
        sp+=list(t*0.4)
        sp+=list(t*0.6)
        sp+=list(t*0.8)
        sp+=list(t*1.0)
        sp+=list(t*1.2)
        sp+=list(t*1.4)
        sp+=list(t*1.6)
        sp+=list(t*1.8)
        sp+=list(t*2.0)
    if moment==3:
        t=[]
        angle=2*pi/3
        t.append([-r*np.cos(angle), r*np.sin(angle)])
        t.append([r*np.cos(angle), r*np.sin(angle)])
        angle=-2*pi/3
        t.append([-r*np.cos(angle), r*np.sin(angle)])
        t.append([r*np.cos(angle), r*np.sin(angle)])
        angle=0
        t.append([r*np.cos(angle), r*np.sin(angle)])
        t.append([-r*np.cos(angle), r*np.sin(angle)])

        t=np.array(t)
        sp+=list(t*0.2)
        sp+=list(t*0.4)
        sp+=list(t*0.6)
        sp+=list(t*0.8)
        sp+=list(t*1)
        sp+=list(t*1.2)
        sp+=list(t*1.4)
        sp+=list(t*1.6)
        sp+=list(t*1.8)
        sp+=list(t*2.0)

    for s in sp:
        # plt.plot(s[0],s[1],"c.")
        ax.streamplot(x, y, Ex, Ey, color="k",
                     linewidth = 0.5,
                     start_points=[s],
                     zorder=0,
                     # minlength=0,
                     integration_direction="both",
                      density=1, arrowstyle='->', arrowsize=1.0)


    # Add filled circles for the charges themselves
    charge_colors = {0: 'C0', 1: 'C3'}
    for q, pos in charges:
        ax.add_artist(Circle(pos, 0.05, color=charge_colors[q>0],zorder=1))


    plt.tick_params( axis='x',          which='both',     bottom=False,    top=False,      labelbottom=False, labelleft=False,)
    plt.tick_params( axis='y',          which='both',     left=False,    top=False,      labelbottom=False, labelleft=False,)

    # plt.axis('off')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    # ax.set_aspect('equal')
    plt.savefig("field-{0}.png".format(moment))
    plt.savefig("field-{0}.pdf".format(moment))

