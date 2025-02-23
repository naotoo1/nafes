import matplotlib.pyplot as plt
import torch


def get_plot(self, cluster, centroids):
    if self.plot_steps:
        for _, v in enumerate(cluster):
            plt.scatter(self.data[v][:, 0], self.data[v][:, 1])
        for cent in centroids:
            plt.scatter(cent[0], cent[1], marker='v', color='black')
            plt.pause(0.3)
            plt.clf()
 
