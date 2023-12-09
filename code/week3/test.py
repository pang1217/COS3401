import numpy as np
import matplotlib.pyplot as plt
 
gamma = [0.5, 0.7, 0.9, 1.3, 1.7, 2.3]
x = np.arange(start=0, stop=256, step=1).astype(np.uint8)
xn = x.astype(np.float16)/np.amax(x)
plt.plot(x, x)
for g in gamma:
    yn = (xn**g)*255
    plt.plot(x, yn.astype(np.uint8))

plt.title("Curve plotted using the given points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
