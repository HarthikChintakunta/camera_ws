import numpy as np
def transform(pos):
    T = np.array([[1, 0, 0],
                  [0,np.cos(np.pi/2),-np.sin(np.pi/2)],
                  [0,np.sin(np.pi/2),np.cos(np.pi/2)]])
    V_t = T@pos
    return V_t

v = np.array([1,2,3])
v_t = transform(v)
print(v_t)