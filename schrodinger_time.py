#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi

h = 6.626e-34
h_bar = h / (2*pi)

E = 2*h_bar

w = E / h_bar


real_list = []
im_list = []

time_steps = np.linspace(0, 10, 50)

for t in (time_steps):
    real = cos(w*t)
    imaginary = -sin(w*t)

    real_list.append(real)
    im_list.append(imaginary)


plt.figure(figsize=(10, 6))

plt.plot(time, real_list, label='Real Component', color='lightskyblue', linewidth=2)
plt.plot(time, im_list, label='Imaginary Component', color='maroon', linewidth=2, linestyle='--')


plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Evolution of the Temporal Solution')
plt.grid(True, linestyle='--')

plt.legend(bbox_to_anchor=(0.85, 0.1))
plt.show()



# In[ ]:




