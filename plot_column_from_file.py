#!/usr/bin/python3
import matplotlib.pyplot as plt


data_from_file = open("error.log", "r").read().splitlines()
try:
    f = [float(x) for x in data_from_file]
    t = range(1, len(f) + 1) # t = [1.. len(f)]
    plt.plot(t, f)
    plt.show()



except:
    print("File is not what expected")

finally:
    print("See you soon!")





