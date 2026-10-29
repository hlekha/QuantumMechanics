#!/usr/bin/env python
# coding: utf-8

# In[1]:


import InfiniteWell_PINN
from InfiniteWell_PINN import _get_tensor, model, _get_psi_prime, _get_second_deriv

#test for shape
x = _get_tensor()
psi = model(x)
dpsi_dx = _get_psi_prime(psi, x)
d2psi_dx2 = _get_second_deriv(dpsi_dx, x)

print(psi.shape)


#test for size of derivatives
print(psi.shape)
print(dpsi_dx.shape)
print(d2psi_dx2.shape)


# In[ ]:




