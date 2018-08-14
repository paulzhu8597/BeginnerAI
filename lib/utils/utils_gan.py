from functools import partial
import numpy
from skimage import transform, filters
from matplotlib import pyplot as plt
plt.switch_backend('agg')

EPS = 1e-66
RESOLUTION = 0.001
num_grids = int(1/RESOLUTION+0.5)

def generate_lut(img):
    density_img = transform.resize(img, (num_grids, num_grids))
    x_accumlation = numpy.sum(density_img, axis=1)
    sum_xy = numpy.sum(x_accumlation)
    y_cdf_of_accumulated_x = [[0., 0.]]
    accumulated = 0
    for ir, i in enumerate(range(num_grids-1, -1, -1)):
        accumulated += x_accumlation[i]
        if accumulated == 0:
            y_cdf_of_accumulated_x[0][0] = float(ir+1)/float(num_grids)
        elif EPS < accumulated < sum_xy - EPS:
            y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), accumulated/sum_xy])
        else:
            break
    y_cdf_of_accumulated_x.append([float(ir+1)/float(num_grids), 1.])
    y_cdf_of_accumulated_x = numpy.array(y_cdf_of_accumulated_x)

    x_cdfs = []
    for j in range(num_grids):
        x_freq = density_img[num_grids-j-1]
        sum_x = numpy.sum(x_freq)
        x_cdf = [[0., 0.]]
        accumulated = 0
        for i in range(num_grids):
            accumulated += x_freq[i]
            if accumulated == 0:
                x_cdf[0][0] = float(i+1) / float(num_grids)
            elif EPS < accumulated < sum_xy - EPS:
                x_cdf.append([float(i+1)/float(num_grids), accumulated/sum_x])
            else:
                break
        x_cdf.append([float(i+1)/float(num_grids), 1.])
        if accumulated > EPS:
            x_cdf = numpy.array(x_cdf)
            x_cdfs.append(x_cdf)
        else:
            x_cdfs.append(None)

    y_lut = partial(numpy.interp, xp=y_cdf_of_accumulated_x[:, 1], fp=y_cdf_of_accumulated_x[:, 0])
    x_luts = [partial(numpy.interp, xp=x_cdfs[i][:, 1], fp=x_cdfs[i][:, 0]) if x_cdfs[i] is not None else None for i in range(num_grids)]

    return y_lut, x_luts

def sample_2d(lut, N):
    y_lut, x_luts = lut
    u_rv = numpy.random.random((N, 2))
    samples = numpy.zeros(u_rv.shape)
    for i, (x, y) in enumerate(u_rv):
        ys = y_lut(y)
        x_bin = int(ys/RESOLUTION)
        xs = x_luts[x_bin](x)
        samples[i][0] = xs
        samples[i][1] = ys

    return samples

class GANDemoVisualizer:

    def __init__(self, title, l_kde=100, bw_kde=5):
        self.title = title
        self.l_kde = l_kde
        self.resolution = 1. / self.l_kde
        self.bw_kde_ = bw_kde
        self.fig, self.axes = plt.subplots(ncols=3, figsize=(13.5, 4))
        self.fig.canvas.set_window_title(self.title)

    def draw(self, real_samples, gen_samples, msg=None, cmap='hot', pause_time=0.05, max_sample_size=500, show=True):
        if msg:
            self.fig.suptitle(msg)
        ax0, ax1, ax2 = self.axes

        self.draw_samples(ax0, 'real and generated samples', real_samples, gen_samples, max_sample_size)
        self.draw_density_estimation(ax1, 'density: real samples', real_samples, cmap)
        self.draw_density_estimation(ax2, 'density: generated samples', gen_samples, cmap)

        if show:
            plt.draw()
            plt.pause(pause_time)

    @staticmethod
    def draw_samples(axis, title, real_samples, generated_samples, max_sample_size):
        axis.clear()
        axis.set_xlabel(title)
        axis.plot(generated_samples[:max_sample_size, 0], generated_samples[:max_sample_size, 1], '.')
        axis.plot(real_samples[:max_sample_size, 0], real_samples[:max_sample_size, 1], 'kx')
        axis.axis('equal')
        axis.axis([0, 1, 0, 1])

    def draw_density_estimation(self, axis, title, samples, cmap):
        axis.clear()
        axis.set_xlabel(title)
        density_estimation = numpy.zeros((self.l_kde, self.l_kde))
        for x, y in samples:
            if 0 < x < 1 and 0 < y < 1:
                density_estimation[int((1-y) / self.resolution)][int(x / self.resolution)] += 1
        density_estimation = filters.gaussian(density_estimation, self.bw_kde_)
        axis.imshow(density_estimation, cmap=cmap)
        axis.xaxis.set_major_locator(plt.NullLocator())
        axis.yaxis.set_major_locator(plt.NullLocator())

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    @staticmethod
    def show():
        plt.show()