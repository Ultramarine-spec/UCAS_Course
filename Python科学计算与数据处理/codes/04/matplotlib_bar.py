# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("china_population.txt")
width = (data[1,0] - data[0,0])*0.4 
plt.figure(figsize=(8,5))
plt.bar(data[:,0]-0.5*width, data[:,1]/1e7, width, color="b", label=u"Male") 
plt.bar(data[:,0]+0.5*width, data[:,2]/1e7, width, color="r", label=u"Female") 
plt.xlim(-width*1.5, 100)
plt.xlabel(u"Age")
plt.ylabel(u"Population(ten million)")
plt.legend()


plt.show()