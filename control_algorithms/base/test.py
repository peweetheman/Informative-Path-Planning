import matplotlib.pyplot as plt
import numpy as np

from control_algorithms.base import dubins_path_planner as plan

max_curvature = 1.0
pose1 = np.array([0,0,0])
pose2 = np.array([5,5,3])
pose3 = np.array([11,3,2])

px1, py1, pangle1, mode1, plength1, u1 = plan.dubins_path_planning(pose1[0], pose1[1], pose1[2], pose2[0], pose2[1], pose2[2], max_curvature)
px2, py2, pangle2, mode2, plength2, u2 = plan.dubins_path_planning(pose1[0], pose1[1], pose1[2], pose3[0], pose3[1], pose3[2], max_curvature)

print(len(px1))
print(len(px2))

plt.plot(px1, py1, color='green')
plt.plot(px2, py2, color='green')
plt.show()

