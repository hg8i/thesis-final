#!/usr/bin/env python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from math import pi

import matplotlib.pyplot as plt

os.popen("rm *png")

fig = plt.figure(figsize=(5,5), dpi=100)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# fig.add_axes(ax)
mu=1

def A(m,r):
    ret = (mu/4/pi)*np.cross(m,r)/abs(r)**3
    return ret

def B(m,r):
    rhat = r/np.linalg.norm(r)
    ret = (mu/4/pi)*(3*rhat*np.dot(m,rhat)-m)/np.linalg.norm(r)**3
    # print np.linalg.norm(r),r**3; quit()
    return ret

res = 1/10
m = np.array([-4,0])
xs=np.arange(-1,1,res)
ys=np.arange(-1,1,res)
dl = 0.01

dm = 0.2
plt.arrow(0,0,m[0]*dm,m[1]*dm,lw=6,head_length=0.08,head_width=0.08,color="r")
for x in xs:
    # if abs(x)<dl: continue
    for y in ys:
        # if abs(y)<dl: continue
        r = np.array([x,y])
        b= B(m,r)/100
        rb=r+b
        if np.linalg.norm(b)<1e-4: continue
        if np.linalg.norm(b)>10: continue
        plt.arrow(r[0],r[1],b[0],b[1],lw=0.4,
                  head_length=0.03,
                  head_width=0.03,
                  color="k",
                 )
        # plt.plot([r[0],r[1]],[rb[0],rb[1]],"k",lw=0.4)
    # break

plt.contour([xs,ys],zs)

plt.ylim([-1,1])
plt.xlim([-1,1])

# plt.savefig(op)
plt.savefig("field.png")


# # Data plot
# plt.clf(); plt.cla()
# fig = plt.figure(figsize=(3,3))
# # ax = plt.subplot(111,aspect = 'equal')
# ax = plt.gca()




# plt.axis('off')
# plt.tight_layout(pad=0.05)
# plt.savefig("data.png",dpi=100)

