import os
import numpy as np
import matplotlib.pyplot as plt
import datetime


class Plot_res(object):
    def __init__(self, plot_save_dir, plot_datainfo, plot_title, plot_xname, plot_yname, enable=False):
        self.data = []
        self.plot_save_dir = plot_save_dir
        self.plot_datainfo = plot_datainfo
        self.plot_title = plot_title
        self.plot_xname = plot_xname
        self.plot_yname = plot_yname
        self.enable = enable
        self.runtime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    def step(self, data):
        assert len(data) == len(self.plot_datainfo)
        self.data.append(data)
        self.save()

    def save(self):
        if self.enable:
            data = np.array(self.data)
            for i in range(len(self.plot_datainfo)):
                x = range(len(self.data))
                plt.plot(x, data[:, i], label=self.plot_datainfo[i])
            plt.title(self.plot_title)
            plt.ylim([0, 0.2])
            plt.xlabel(self.plot_xname)
            plt.ylabel(self.plot_yname)
            if not os.path.exists(self.plot_save_dir):
                os.makedirs(self.plot_save_dir)
            plt.legend()
            plt.savefig(os.path.join(self.plot_save_dir, '{}.jpg'.format(self.runtime)))
            plt.close()


if __name__ == "__main__":
    print(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # testplot = Plot_res('./tmp', ['bxbx'], 'title', 'epoch', 'ETS', True)
    # for i in range(15):
    #     testplot.step([i * 2])
