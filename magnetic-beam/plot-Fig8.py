import numpy as np
import matplotlib.pyplot as plt


plt.plot([], [], label='FEM', color='black')
plt.scatter([], [], label='Hex LSM', marker='d', color='black')
plt.scatter([], [], label='Tet MSM', marker='x', color='black')
plt.scatter([], [], label=r'$\delta_x$', marker='o', edgecolors='tab:orange', facecolors='tab:orange')
plt.scatter([], [], label=r'$\delta_z$', marker='o', edgecolors='tab:blue', facecolors='tab:blue')

PK2 = np.loadtxt('Mz-deflection-fenicsx-PK2.txt')
plt.plot(PK2[:,0], -PK2[:,1], color='tab:orange')
plt.plot(PK2[:,0], PK2[:,2], color='tab:blue')

## 4 layer Hex LSM
lsm = np.loadtxt('Mz-deflection-LSM.txt')
# get the index for the last 5 for each unique B
idx = np.unique(lsm[:, 0], return_index=True)[1]
lsmlast5 = np.zeros((len(idx)*5, 3), dtype=float)
for i, idx in enumerate(np.roll(idx,-1)):
    j, k = i*5, (i+1)*5
    if idx-5 > 0:
        lsmlast5[j:k] = lsm[idx-5:idx]
    else:
        lsmlast5[j:k] = lsm[-5:]

zB = lsmlast5[:, 0]
zdx = np.abs(lsmlast5[:, 1])
zdz = lsmlast5[:, 2]
plt.scatter(zB, zdx, marker='d', color='tab:orange')
plt.scatter(zB, zdz, marker='d', color='tab:blue')

## 4 layer Tet MSM
msm = np.loadtxt('dump/Yan-N4-Mz-deflection.txt')
# get the index for the last unique B
idx = np.roll(np.unique(msm[:,0], return_index=True)[1], -1) - 1
plt.plot(msm[idx,0], -msm[idx,1], marker='x', color='tab:orange', linestyle=':')
plt.plot(msm[idx,0], msm[idx,2], marker='x', color='tab:blue', linestyle=':')


plt.legend(loc='upper left', fontsize=14, markerscale=1)
plt.xlim(0, 100)
plt.ylim(0, 1)

plt.tick_params(labelsize=14)

# plt.xlabel(r'$M (\nabla B)_{zz} AL^3 / EI$', fontsize=14)
plt.xlabel(r'$\lambda_m^\nabla$', fontsize=14)
plt.ylabel(r'$\delta/L$', fontsize=14)
plt.tight_layout()

# plt.savefig("Fig8-FEM-vs-MSM.png", dpi=600)

plt.show()