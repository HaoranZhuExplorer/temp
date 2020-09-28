from collections import OrderedDict
import matplotlib.pyplot as plt

from plot import plot_curve

plots = OrderedDict()
#plots['jpg'] = ('jpeg.csv', {})
#plots['jpg2000'] = ('jpeg2000.csv', {})
#plots['diff_jpg'] = ('diff_jpeg.csv', {})
plots['guetzli'] = ('guetzli.csv', {})
#plots['bpg'] = ('bpg.csv', {})



#ratedistortion.load_data(plots, "./data/rgb/clic512")
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

plot_curve(plots, ax, dirname="/Users/Haoran/Desktop/data/raw512", metric='ssim')
plt.show()