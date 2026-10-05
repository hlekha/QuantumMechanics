#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi

E = 1
h = 6.626e-34
h_bar = h / 2*pi
w = E / h_bar


real_list = []
im_list = []

time_steps = np.arange(0, 10)

for t in range(len(time_steps)):
    real = cos(w*t)
    imaginary = -sin(w*t)

    real_list.append(real)
    im_list.append(imaginary)


plt.figure(figsize=(8,6))
scatter = plt.scatter(real_list, im_list, c=time_steps, cmap='viridis', s=60, edgecolor='k', zorder=5)


cbar = plt.colorbar(scatter)
cbar.set_label('Time')

plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Evolution of the Temporal Solution')
plt.grid(True, linestyle='--')

plt.show()


# In[ ]:




