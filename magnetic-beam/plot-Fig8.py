import numpy as np
import matplotlib.pyplot as plt

PK2 = np.loadtxt('Mz-deflection-fenicsx-PK2.txt')

plt.scatter([], [], label='FEM', marker='o', edgecolors='black', facecolors='none')
plt.scatter([], [], label='MSM', marker='x', color='black')
plt.scatter([], [], label=r'$\delta_x$', marker='o', edgecolors='tab:orange', facecolors='tab:orange')
plt.scatter([], [], label=r'$\delta_y$', marker='o', edgecolors='tab:blue', facecolors='tab:blue')


plt.scatter(PK2[:,0], -PK2[:,1], marker='o', facecolors='none', edgecolors='tab:orange')
plt.scatter(PK2[:,0], PK2[:,2], marker='o', facecolors='none', edgecolors='tab:blue')

msm = np.loadtxt('Yan-N4-Mz-deflection.txt')

# get the index for the last unique B
idx = np.roll(np.unique(msm[:,0], return_index=True)[1], -1) - 1

plt.scatter(msm[idx,0], -msm[idx,1], marker='x', facecolors='tab:orange')
plt.scatter(msm[idx,0], msm[idx,2], marker='x', facecolors='tab:blue')


plt.legend(loc='upper left', fontsize=14, markerscale=1)
plt.xlim(0, 100)

plt.tick_params(labelsize=14)

plt.ylim(0, 1)

# plt.xlabel(r'$M (\nabla B)_{zz} AL^3 / EI$', fontsize=14)
plt.xlabel(r'$\lambda_m^\nabla$', fontsize=14)
plt.ylabel(r'$\delta/L$', fontsize=14)
plt.tight_layout()

# plt.savefig("Fig8-FEM-vs-MSM.png", dpi=600)

plt.show()