import os
import numpy as np
from loguru import logger
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
import lpips

import glob, os
import pandas as pd
import numpy as np
from skimage.measure import compare_ssim, compare_psnr
import tqdm
from sewar.full_ref import msssim
import imageio
import io

from jpeg_helpers import JPEGMarkerStats
#os.system('export DYLD_LIBRARY_PATH="/Users/Haoran/Documents/workspace/file/ImageMagick-7.0.10/lib/" && magick convert -quality 50 -define jpeg:q-table=/Users/Haoran/Desktop/quantization-table.xml /Users/Haoran/Downloads/raw512/md0d3ca2e738.png /Users/Haoran/Downloads/raw512/image.jpeg')

from lpips_2imgs import compute_perceptual_similarity
'''
os.chdir("/Users/Haoran/Downloads/raw512")

for file in glob.glob('*.png'):
    # remove .png extension
    file = file[:-4]
    print(file)
'''

def get_diff_jpeg_df(directory, write_files=False, effective_bytes=True, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for JPEG. The result is saved
    as a CSV file in the source directory. If the file exists, the DF is loaded and returned.
    Files are saved as JPEG using imageio.
    """

    # Get trade-off for JPEG
    quality_levels = np.arange(95, 5, -5)
    df_jpeg_path = os.path.join(directory, 'diff_jpeg.csv')

    os.chdir(directory)
    files = [file for file in glob.glob('*.png')]

    if os.path.isfile(df_jpeg_path) and not force_calc:
        print('Restoring JPEG stats from {}'.format(df_jpeg_path))
        df = pd.read_csv(df_jpeg_path, index_col=False)
    else:
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'msssim', 'msssim_db', 'bytes', 'bpp'])

        with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='JPEG') as pbar:

            for image_id, filename in enumerate(files):

                for qi, q in enumerate(quality_levels):

                    # Compress images and get effective bytes (only image data - no headers)
                    os.system('export DYLD_LIBRARY_PATH="/Users/Haoran/Documents/workspace/file/ImageMagick-7.0.10/lib/" && magick convert -quality '+str(q)+' -define jpeg:q-table=/Users/Haoran/Desktop/quantization-table.xml '+directory+'/'+filename+' '+directory+'/'+filename+"_"+str(q)+'_compressed.jpeg')
                    #os.system('export DYLD_LIBRARY_PATH="/Users/Haoran/Documents/workspace/file/ImageMagick-7.0.10/lib/" && magick convert -quality ' + str(q) + ' -define jpeg:q-table=/Users/Haoran/Desktop/quantization-table_standard.xml ' + directory + '/' + filename + ' ' + directory + '/' + filename + "_" + str(q) + '_compressed.jpeg')
                    if effective_bytes:
                        with open(directory+'/'+filename+"_"+str(q)+'_compressed.jpeg', 'rb') as fh:
                            buf = io.BytesIO(fh.read())
                        image_bytes = JPEGMarkerStats(buf.getvalue()).get_effective_bytes()
                    else:
                        image_bytes = os.path.getsize(directory+'/'+filename+"_"+str(q)+'_compressed.jpeg')
                    image = imageio.imread(directory+'/'+filename).astype(np.float) / (2**8 - 1)
                    image_compressed = imageio.imread(directory+'/'+filename+"_"+str(q)+'_compressed.jpeg').astype(np.float) / (2**8 - 1)
                    image_compressed_path = directory + '/' + filename + "_" + str(q) + '_compressed.jpeg'
                    image_path = directory + '/' + filename
                    perceptual_similarity = os.system("python3 /Users/Haoran/Documents/workspace/rate_distortion/src/lpips_2imgs.py -p0 " + directory + '/' + filename + "_" + str(q) + '_compressed.jpeg' + ' -p1 ' + directory + '/' + filename)

                    os.remove(directory+'/'+filename+"_"+str(q)+'_compressed.jpeg')
                    msssim_value = msssim(image, image_compressed, MAX=1).real

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'jpeg',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
                                    'msssim': msssim_value,
                                    'msssim_db': -10 * np.log10(1 - msssim_value),
                                    'perceptual similarity': perceptual_similarity,
                                    'bytes': image_bytes,
                                    'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                    }, ignore_index=True)

                    pbar.set_postfix(image_id=image_id, quality=q)
                    pbar.update(1)
        df.to_csv(os.path.join(directory, 'diff_jpeg.csv'), index=False)


    return df


def get_guetzli_df(directory, write_files=False, effective_bytes=True, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for JPEG. The result is saved
    as a CSV file in the source directory. If the file exists, the DF is loaded and returned.
    Files are saved as JPEG using imageio.
    """

    # Get trade-off for JPEG
    quality_levels = np.arange(99, 83, -2)
    df_jpeg_path = os.path.join(directory, 'guetzli.csv')

    os.chdir(directory)
    files = [file for file in glob.glob('*.png')]

    if os.path.isfile(df_jpeg_path) and not force_calc:
        print('Restoring JPEG stats from {}'.format(df_jpeg_path))
        df = pd.read_csv(df_jpeg_path, index_col=False)
    else:
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'msssim', 'msssim_db', 'bytes', 'bpp'])

        with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='JPEG') as pbar:

            for image_id, filename in enumerate(files):
                print(filename)

                for qi, q in enumerate(quality_levels):
                    print(q)
                    # Compress images and get effective bytes (only image data - no headers)
                    os.system('/home/Haoran/guetzli/bin/Release/./guetzli --quality '+str(q)+' '+directory+'/'+filename+' '+directory+'/'+filename+"_"+str(q)+'_compressed.jpg')
                    if effective_bytes:
                        with open(directory+'/'+filename+"_"+str(q)+'_compressed.jpeg', 'rb') as fh:
                            buf = io.BytesIO(fh.read())
                        print(buf.getvalue())
                        image_bytes = JPEGMarkerStats(buf.getvalue()).get_effective_bytes()

                    else:
                        image_bytes = os.path.getsize(directory+'/'+filename+"_"+str(q)+'_compressed.jpg')
                    image = imageio.imread(directory+'/'+filename).astype(np.float) / (2**8 - 1)
                    image_compressed = imageio.imread(directory+'/'+filename+"_"+str(q)+'_compressed.jpg').astype(np.float) / (2**8 - 1)
                    image_compressed_path = directory + '/' + filename + "_" + str(q) + '_compressed.jpg'
                    image_path = directory + '/' + filename
                    perceptual_similarity = compute_perceptual_similarity(image_path, image_compressed_path)

                    os.remove(directory+'/'+filename+"_"+str(q)+'_compressed.jpg')
                    msssim_value = msssim(image, image_compressed, MAX=1).real

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'guetzli',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
                                    'msssim': msssim_value,
                                    'msssim_db': -10 * np.log10(1 - msssim_value),
                                    'perceptual similarity': perceptual_similarity,
                                    'bytes': image_bytes,
                                    'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                    }, ignore_index=True)

                    pbar.set_postfix(image_id=image_id, quality=q)
                    pbar.update(1)
                    break
                break
        df.to_csv(os.path.join(directory, 'guetzli.csv'), index=False)


    return df
def load_data(plots, dirname):
    """
    Returns data frames with numerical results for specified codecs [and settings]
    Example definition (can be both a list or a dictionary):
    plots = OrderedDict()
    plots['jpg'] = ('jpeg.csv', {})
    plots['jp2'] = ('jpeg2000.csv', {})
    plots['bpg'] = ('bpg.csv', {})
    plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})
    Tuple structure: (filename, data filtering conditions - dict {column: value})
    """

    # Load all needed tables and setup legend labels
    labels = []
    df_all = []

    if isinstance(plots, list):
        for filename, selectors in plots:
            labels.append(os.path.splitext(filename)[0])
            df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
            for k, v in selectors.items():
                if isinstance(v, str) and '*' in v:
                    df = df[df[k].str.match(v)]
                else:
                    df = df[df[k] == v]
            if len(df) == 0:
                raise (ValueError('No rows matched for column {}'.format(k)))
            df_all.append(df)

    elif isinstance(plots, dict):
        for key, (filename, selectors) in plots.items():
            labels.append(key)
            df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
            for k, v in selectors.items():
                if isinstance(v, str) and '*' in v:
                    df = df[df[k].str.match(v)]
                else:
                    df = df[df[k] == v]
            if len(df) == 0:
                raise (ValueError('No rows matched for column {}'.format(k)))
            df_all.append(df)
    else:
        raise ValueError('Unsupported plot definition!')

    return df_all, labels


def setup_plot(metric):
    if metric == 'psnr':
        y_min = 25
        y_max = 45
        metric_label = 'PSNR [dB]'

    elif metric == 'msssim_db':
        y_min = 10
        y_max = 32
        metric_label = 'MS-SSIM [dB]'

    elif metric == 'ssim':
        y_min = 0.8
        y_max = 1
        metric_label = 'SSIM'

    elif metric == 'msssim':
        y_min = 0.9
        y_max = 1
        metric_label = 'MS-SSIM'
    else:
        raise ValueError('Unsupported metric!')

    return y_min, y_max, metric_label


def setup_fit(metric):
    # Define a parametric model for the trade-off curve
    if metric in {'ssim', 'msssim'}:
        # These bounds work well for baseline fitting
        fit_bounds = ([1e-4, 1e-2, -3, -0.5], [5, 15, 5, 0.5])
        # These bounds work better for optimized DCN codecs - there are some weird outliers in the data
        # fit_bounds = ([0.1, 1e-5, -1, 0], [3, 10, 7, 0.1])

        def func(x, a, b, c, d):
            return 1 / (1 + np.exp(- b * x ** a + c)) - d
    else:
        # These bounds work well for baseline fitting
        fit_bounds = ([1e-4, 1e-5, 1e-2, -50], [100, 100, 3, 50])
        # These bounds work better for optimized DCN codecs - there are some weird outliers in the data
        # fit_bounds = ([1e-4, 1, 1e-2, -20], [20, 50, 1, 20])

        def func(x, a, b, c, d):
            return a * np.log(np.clip(b * x ** c + d, a_min=1e-9, a_max=1e9))

    return func, fit_bounds

def plot_curve(plots, axes,
               dirname='',
               images=[],
               plot='fit',
               draw_markers=None,
               metric='psnr',
               title=None,
               add_legend=True,
               marker_legend=True,
               baseline_count=3,
               update_ylim=False):

    # Parse input parameters
    draw_markers = draw_markers if draw_markers is not None else len(images) == 1
    plot = 'fit'

    df_all, labels = load_data(plots, dirname)

    if len(images) == 0:
        images = df_all[0]['image_id'].unique().tolist()

    # Plot setup
    func, fit_bounds = setup_fit(metric)
    y_min, y_max, metric_label = setup_plot(metric)

    # Select measurements for specific images, if specified
    for dfc in df_all:
        if len(images) > 0:
            dfc['selected'] = dfc['image_id'].apply(lambda x: x in images)
        else:
            dfc['selected'] = True

    # Setup drawing styles
    styles = [['r-', 'rx'], ['b--', 'b+'], ['k:', 'k2'], ['g-', 'gx'], ['m-', 'gx'], ['m--', 'gx'], ['m-.', 'gx'], ['m:', 'gx']]
    avg_markers = ['', '', '', 'o', 'o', '2', '+', 'x', '^', '.']

    # To retain consistent styles across plots, adjust the lists based on the number of baseline methods
    if baseline_count < 3:
        styles = styles[(3 - baseline_count):]
        avg_markers = avg_markers[(3 - baseline_count):]

    # Iterate over defined plots and draw data accordingly
    for index, dfc in enumerate(df_all):

        x = dfc.loc[dfc['selected'], 'bpp'].values
        y = dfc.loc[dfc['selected'], metric].values

        X = np.linspace(max([0, x.min() * 0.9]), min([5, x.max() * 1.1]), 256)

        if plot == 'fit':
            # Fit individual images to a curve, then average the curves

            Y = np.zeros((len(images), len(X)))
            mse_l = []

            for image_no, image_id in enumerate(images):

                x = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), 'bpp'].values
                y = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), metric].values

                # Allow for larger errors for lower SSIM values
                if metric in ['ssim', 'msssim']:
                    sigma = np.abs(1 - y).reshape((-1,))
                else:
                    sigma = np.ones_like(y).reshape((-1,))

                try:
                    popt, pcov = curve_fit(func, x, y, bounds=fit_bounds, maxfev=10000, sigma=sigma)
                    y_est = func(x, *popt)
                    mse = np.mean(np.power(y - y_est, 2))
                    mse_l.append(mse)
                    if mse > 0.5:
                        logger.warning('WARNING Large MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))

                except RuntimeError:
                    logger.error(f'{labels[index]} image ={image_id}, bpp ={x} y ={y}')

                Y[image_no] = func(X, *popt)

            if len(images) > 1:
                logger.info('Fit summary - MSE for {} av={:.2f} max={:.2f}'.format(labels[index], np.mean(mse_l), np.max(mse_l)))

            yy = np.nanmean(Y, axis=0)
            axes.plot(X, yy, styles[index][0], label=labels[index] if add_legend else None)
            y_min = min([y_min, min(yy)]) if update_ylim else y_min

        elif plot == 'aggregate':
            # For each quality level (QF, #channels) find the average quality level
            dfa = dfc.loc[dfc['selected']]

            if 'n_features' in dfa:
                dfg = dfa.groupby('n_features')
            else:
                dfg = dfa.groupby('quality')

            x = dfg.mean()['bpp'].values
            y = dfg.mean()[metric].values

            axes.plot(x, y, styles[index][0], label=labels[index] if add_legend else None, marker=avg_markers[index], alpha=0.65)
            y_min = min([y_min, min(y)]) if update_ylim else y_min

        elif plot == 'none':
            pass

        else:
            raise ValueError('Unsupported plot type!')

        if draw_markers:

            if 'entropy_reg' in dfc:

                # No need to draw legend if multiple DCNs are plotted
                detailed_legend = 'full' if marker_legend and index == baseline_count else False

                style_mapping = {}

                if 'n_features' in dfc and len(dfc['n_features'].unique()) > 1:
                    style_mapping['hue'] = 'n_features'

                if 'entropy_reg' in dfc and len(dfc['entropy_reg'].unique()) > 1:
                    style_mapping['size'] = 'entropy_reg'

                if 'quantization' in dfc and len(dfc['quantization'].unique()) > 1:
                    style_mapping['style'] = 'quantization'

                sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                palette="Set2", ax=axes, legend=detailed_legend,
                                **style_mapping)

            else:
                axes.plot(x, y, styles[index][1], alpha=10 / (sum(dfc['selected'])))

    n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())

    title = '{} : {}'.format(
        title if title is not None else os.path.split(dirname)[-1],
        '{} images'.format(n_images) if n_images > 1 else dfc.loc[dfc['selected'], 'filename'].unique()[0].replace('.png', '')
    )

    # Fixes problems with rendering using the LaTeX backend
    if add_legend:
        for t in axes.legend().texts:
            t.set_text(t.get_text().replace('_', '-'))

    axes.set_xlim([-0.1, 3.1])
    axes.set_ylim([y_min * 0.99, y_max])
    axes.set_title(title)
    axes.set_xlabel('Effective bpp')
    axes.set_ylabel(metric_label)
