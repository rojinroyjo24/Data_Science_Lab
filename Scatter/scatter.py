import matplotlib.pyplot as plt
import numpy as np
x=np.array(["jan","feb","march","april","may"])
affordable=np.array([2,4,6,8,10])
luxury=np.array([5,10,15,20,25])
superlux=np.array([10,20,30,40,50])
plt.scatter(x,affordable)
plt.scatter(x,luxury)
plt.scatter(x,superlux)
plt.xlabel("month of year")
plt.ylabel("sales")
plt.title("sales data")
plt.show()
plt.plot(x,y,ls="-.",color="r",marker="*",markerfacecolor="g",markersize=20)
