import numpy as np
import matplotlib.pyplot as plt
from semsetup import *

p1 = 5
p2 = 8

ah1,bh1,ch1,dh1,z1,w1 = semhat(p1)
ah2,bh2,ch2,dh2,z2,w2 = semhat(p2)

pts1y = z1
pts1x = z1-1
pts2y = z2
pts2x = z2+1
[X1,Y1] = np.meshgrid(pts1x,pts1y)
[X2,Y2] = np.meshgrid(pts2x,pts2y)

plt.plot(X1,Y1,marker=None,linestyle=':',color='k')
plt.plot(X1.T,Y1.T,marker=None,linestyle=':',color='k')
plt.plot(X2,Y2,marker=None,linestyle=':',color='k')
plt.plot(X2.T,Y2.T,marker=None,linestyle=':',color='k')
plt.axis("equal")
plt.axis("off")
plt.annotate("$\\Omega_1$",xy=(-1,1.1), fontsize=24)
plt.annotate("$\\Omega_2$",xy=(1,1.1), fontsize=24)
plt.annotate("$\\Gamma$",xy=(-0.18,0), fontsize=24)
plt.show()
#plt.savefig("../poster/omega1omega2.png", bbox_inches='tight', pad_inches=0)
