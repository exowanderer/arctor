import batman
import joblib
import logging
import numpy as np
import os
import pandas as pd
import warnings
import multiprocessing as mp

from astropy.io import fits
from astropy.modeling.models import Gaussian1D, Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
# from astropy.modeling.fitting import SLSQPLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from functools import partial
from glob import glob
from matplotlib import pyplot as plt
from photutils import RectangularAperture, RectangularAnnulus
from photutils import aperture_photometry
from scipy.optimize import minimize
from statsmodels.robust import scale as sc
from time import time
from tqdm import tqdm


def debug_message(message, end='\n'):
    print(f'[DEBUG] {message}', end=end)


def warning_message(message, end='\n'):
    print(f'[WARNING] {message}', end=end)


def info_message(message, end='\n'):
    print(f'[INFO] {message}', end=end)


def center_one_trace(kcol, col, fitter, stddev, y_idx, inds):
    model = Gaussian1D(amplitude=col.max(),
                       mean=y_idx, stddev=stddev)

    results = fitter(model, inds, col)
    return kcol, results, fitter


def fit_one_slopes(kimg, means, fitter, y_idx, slope_guess=2.0 / 466):
    model = Linear1D(slope=slope_guess, intercept=y_idx)

    inds = np.arange(len(means))
    inds = inds - np.median(inds)

    results = fitter(model, inds, means)

    return kimg, results, fitter


def cosmic_ray_flag_simple(image_, nSig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_Med = np.median(row)
        row_Std = np.std(row)
        cosmic_rays_[k] += abs(row - row_Med) > nSig * row_Std
        image_[k][cosmic_rays_[k]] = row_Med

    return image_, cosmic_rays_


def cosmic_ray_flag_rolling(image_, nSig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_rMed = pd.Series(row).rolling(window).median()
        row_rStd = pd.Series(row).rolling(window).std()
        cosmic_rays_[k] += abs(row - row_rMed) > nSig * row_rStd
        image_[k][cosmic_rays_[k]] = row_rMed[cosmic_rays_[k]]

    return image_, cosmic_rays_


class HSTUVISTimeSeries(object):

    def __init__(self, planet_name='planetName', data_dir='./',
                 working_dir='./', file_type='flt.fits'):
        info_message('Initializing Instance of the `HSTUVISTimeSeries` Object')
        self.planet_name = planet_name
        self.data_dir = data_dir
        self.working_dir = working_dir
        self.file_type = file_type
        self.configure_matplotlib()

    def cosmic_ray_flag(self, image_, nSig=5, window=7):
        return cosmic_ray_flag_simple(image_, nSig=nSig, window=window)

    def clean_cosmic_rays(self, nSig=5, window=7):
        info_message('Flagging Cosmic Rays using `Temporal Simple` Technique')
        return self.clean_cosmic_rays_temporal_simple(nSig=nSig, window=window)

    def clean_cosmic_rays_temporal_rolling(self, nSig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        for krow in tqdm(range(self.width)):
            for kcol in range(self.height):
                val = self.image_stack[:, kcol, krow]
                val_Med = pd.Series(val).rolling(window).median()
                val_Std = pd.Series(val).rolling(window).std()
                mask = abs(val - val_Med) > nSig * val_Std
                self.cosmic_rays[:, kcol, krow] = mask
                self.image_stack[mask, kcol, krow] = val_Med[mask]

    def clean_cosmic_rays_temporal_simple(self, nSig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        for krow in tqdm(range(self.width)):
            for kcol in range(self.height):
                val = self.image_stack[:, kcol, krow]
                val_Med = np.median(val)
                val_Std = np.std(val)
                mask = abs(val - val_Med) > nSig * val_Std
                self.cosmic_rays[:, kcol, krow] = mask
                self.image_stack[mask, kcol, krow] = val_Med

    def clean_cosmic_rays_spatial(self, nSig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        for k, image_ in tqdm(enumerate(self.image_stack),
                              total=self.n_images):

            image_clean_, cosmic_rays_ = self.cosmic_ray_flag(image_,
                                                              nSig=nSig,
                                                              window=window)

            self.image_stack[k] = image_clean_
            self.cosmic_rays[k] = cosmic_rays_

    def center_all_traces(self, stddev=2, plot_verbose=False):
        info_message('Computing the Center of the Trace')
        if not hasattr(self, 'height') or not hasattr(self, 'width'):
            self.height, self.width = self.image_shape

        inds = np.arange(self.height)
        partial_center_one_trace = partial(center_one_trace,
                                           fitter=LevMarLSQFitter(),
                                           stddev=stddev,
                                           y_idx=self.y_idx,
                                           inds=inds)

        self.center_traces = {}
        for kimg, image in tqdm(enumerate(self.image_stack),
                                total=self.n_images):
            self.center_traces[kimg] = {}

            start = time()
            info_message(f'Starting Multiprocess for Image {kimg}')

            zipper = zip(np.arange(self.width), image.T)

            pool = mp.Pool(mp.cpu_count() - 1)
            center_traces_ = pool.starmap(partial_center_one_trace, zipper)

            pool.close()
            pool.join()

            # center_traces_ = [partial_center_one_trace(
            #     *entry) for entry in zipper]
            rtime = time() - start
            info_message(f'Center computing Image {kimg} took {rtime} seconds')

            for kcol, results, fitter in center_traces_:
                self.center_traces[kimg][kcol] = {}
                self.center_traces[kimg][kcol]['results'] = results
                self.center_traces[kimg][kcol]['fitter'] = fitter

            if False and plot_verbose:
                self.plot_trace_peaks(image)

    def fit_trace_slope(self, stddev=2, plot_verbose=False):
        info_message('fitting a slope to the Center of the Trace')
        if not hasattr(self, 'center_traces'):
            self.center_all_traces(stddev=stddev, plot_verbose=plot_verbose)

        trace_width = self.x_right - self.x_left
        self.gaussian_centers = np.zeros((self.n_images, self.width))
        for kimg, val0 in self.center_traces.items():
            for kcol, val1 in val0.items():
                self.gaussian_centers[kimg][kcol] = val1['results'].mean.value

        partial_fit_slp = partial(fit_one_slopes,
                                  y_idx=self.y_idx,
                                  fitter=LinearLSQFitter(),
                                  slope_guess=2.0 / 466)

        zipper = zip(np.arange(self.n_images),
                     self.gaussian_centers[:, self.x_left:self.x_right])

        slopInts = [partial_fit_slp(*entry) for entry in tqdm(zipper)]

        self.image_line_fits = {}
        for kimg, results, fitter in slopInts:
            self.image_line_fits[kimg] = {}
            self.image_line_fits[kimg]['results'] = results
            self.image_line_fits[kimg]['fitter'] = fitter

        self.image_slopes = np.ones(self.n_images)
        self.image_intercepts = np.ones(self.n_images)
        for kimg, val in self.image_line_fits.items():
            self.image_slopes[kimg] = val['results'].slope.value
            self.image_intercepts[kimg] = val['results'].intercept.value

        if plot_verbose:
            useful = gaussian_centers.T[self.x_left:self.x_right]
            fig, ax = plt.subplots()
            med_trace = np.median(useful, axis=0) * 0
            ax.plot(useful - med_trace)

    def plot_trace_peaks(self, image):
        gauss_means = np.zeros(image_shape[1])
        for key, val in self.center_traces.items():
            gauss_means[key] = val['results'].mean.value

        norm = simple_norm(image, 'sqrt', percent=99)
        plt.imshow(image, norm=norm)
        plt.plot(np.arange(image_shape[1]), gauss_means,
                 'o', color='C1', ms=1)

        plt.xlim(0, image_shape[1])
        plt.ylim(0, image_shape[0])
        plt.tight_layout()
        plt.axis('off')
        plt.waitforbuttonpress()

    @staticmethod
    def plot_apertures(image, aperture,
                       inner_annular=None, outer_annular=None):
        norm = simple_norm(image, 'sqrt', percent=99)

        plt.imshow(image, norm=norm)

        aperture.plot(color='white', lw=2)

        if inner_annular is not None:
            inner_annular.plot(color='red', lw=2)

        if outer_annular is not None:
            outer_annular.plot(color='violet', lw=2)

        plt.axis('off')
        plt.tight_layout()
        plt.waitforbuttonpress()

    def compute_sky_background(self, positions=None,
                               inner_width=None, outer_width=None,
                               inner_height=None, outer_height=None,
                               thetas=None, plot_verbose=False, done_it=False):
        '''
            Run photometry for a specifc set of rectangles

            Parameters
            ----------
            positions (nD-array; 2 x n_images): (xcenter, ycenter)
            widths (nD-array; 3 x n_images):
                    (aperture, inner_annular, outer_annular)
            heights (nD-array; 3 x n_images):
                    (aperture, inner_annular, outer_annular)
        '''

        n_images = self.n_images  # convenience for minimizing command lengths

        x_width = self.x_right - self.x_left

        if positions is None:
            positions = [[self.width // 2, self.y_idx]] * n_images

        if inner_width is None:
            inner_width = 75
        if outer_width is None:
            outer_width = 150

        if inner_height is None:
            inner_height = 225
        if outer_height is None:
            outer_height = 350

        if thetas is None:
            thetas = [0] * n_images

        inner_width = x_width + inner_width
        outer_width = x_width + outer_width

        sky_bgs = np.zeros(n_images)

        self.outer_annulars = []
        self.inner_annulars = []

        zipper = enumerate(zip(self.image_stack, positions, thetas))

        for k, (image, pos, theta) in tqdm(zipper, total=n_images):
            outer_annular = RectangularAperture(
                pos, outer_width, outer_height, theta)
            inner_annular = RectangularAperture(
                pos, inner_width, inner_height, theta)

            self.outer_annulars.append(outer_annular)
            self.inner_annulars.append(inner_annular)

            inner_table = aperture_photometry(image, inner_annular,
                                              method='subpixel', subpixels=32)
            outer_table = aperture_photometry(image, outer_annular,
                                              method='subpixel', subpixels=32)

            inner_flux = inner_table['aperture_sum'][0]
            outer_flux = outer_table['aperture_sum'][0]
            background_area = outer_annular.area - inner_annular.area
            sky_bgs[k] = (outer_flux - inner_flux) / background_area

        self.sky_bgs = sky_bgs

    def do_phot(self, positions=None,
                aper_width=None, aper_height=None,
                thetas=None, plot_verbose=False, done_it=False):
        '''
            Run photometry for a specifc set of rectangles

            Parameters
            ----------
            positions (nD-array; 2 x n_images): (xcenter, ycenter)
            aper_width (float): width of photometry aperture
            aper_height (float): height of photometry aperture
        '''

        n_images = self.n_images  # convenience for minimizing command lengths
        x_width = self.x_right - self.x_left

        if positions is None:
            positions = [[self.width // 2, self.y_idx]] * n_images

        if aper_width is None:
            aper_width = 50

        if aper_height is None:
            aper_height = 200

        if thetas is None:
            thetas = [0] * n_images

        aper_width = x_width + aper_width

        if not hasattr(self, 'fluxes'):
            self.fluxes = {}
            self.fluxes['apertures'] = {}
            self.fluxes['positions'] = {}
            self.fluxes['aper_width'] = {}
            self.fluxes['aper_height'] = {}
            self.fluxes['thetas'] = {}
            self.fluxes['fluxes'] = {}
            self.fluxes['errors'] = {}

        fluxes_ = np.zeros(n_images)
        errors_ = np.zeros(n_images)

        apertures_ = []

        zipper = enumerate(zip(self.image_stack, positions, thetas))

        for k, (image, pos, theta) in tqdm(zipper, total=n_images):
            aperture = RectangularAperture(pos, aper_width, aper_height, theta)
            apertures_.append(aperture)

            if plot_verbose and not done_it:
                aperture = apertures_[k]
                inner_annular = self.inner_annulars[k]
                outer_annular = self.outer_annulars[k]
                plot_apertures(image, aperture, inner_annular, outer_annular)
                done_it = True

            image_table = aperture_photometry(image, aperture,
                                              method='subpixel',
                                              subpixels=32)

            background = self.sky_bgs[k] * aperture.area
            fluxes_[k] = image_table['aperture_sum'][0] - background

        errors_ = np.sqrt(fluxes_)  # explicitly state Poisson noise limit

        id_ = f'{np.random.randint(1e7):0>7}'
        self.fluxes['apertures'][id_] = apertures_
        self.fluxes['positions'][id_] = positions
        self.fluxes['aper_width'][id_] = aper_width
        self.fluxes['aper_height'][id_] = aper_height
        self.fluxes['thetas'][id_] = thetas
        self.fluxes['fluxes'][id_] = fluxes_
        self.fluxes['errors'][id_] = errors_

    def do_multi_phot(self, aper_widths, aper_heights,
                      position=None, theta=None):

        info_message('Beginning Multi-Aperture Photometry')

        x_width = self.x_right - self.x_left

        pos = [self.width // 2, self.y_idx] if position is None else position
        theta = 0 if theta is None else theta

        aper_widths = x_width + aper_widths

        if not hasattr(self, 'fluxes'):
            self.fluxes = {}
            self.fluxes['apertures'] = {}
            self.fluxes['positions'] = {}
            self.fluxes['aper_width'] = {}
            self.fluxes['aper_height'] = {}
            self.fluxes['thetas'] = {}
            self.fluxes['fluxes'] = {}
            self.fluxes['errors'] = {}

        zipper = enumerate(zip(aper_widths, aper_heights))
        info_message('Creating Apertures')
        apertures = []
        for aper_height in aper_heights:
            for aper_width in aper_widths:
                aperture = RectangularAperture(
                    pos, aper_width, aper_height, theta)
                apertures.append(aperture)

        partial_aper_phot = partial(aperture_photometry, apertures=apertures,
                                    method='subpixel', subpixels=32)

        start = time()
        info_message('Computing Aperture Photomery per Image')
        pool = mp.Pool(mp.cpu_count() - 1)
        aper_phots = pool.starmap(partial_aper_phot, zip(self.image_stack))
        pool.close()
        pool.join()

        aper_df = aper_phots[0].to_pandas()

        for kimg in aper_phots[1:]:
            aper_df = pd.concat([aper_df, kimg.to_pandas()])

        photometry_df = aper_df.reset_index().drop(['index', 'id'], axis=1)

        if not hasattr(self, 'photometry_df'):
            self.photometry_df = photometry_df
        else:
            self.photometry_df = pd.concat([self.photometry_df, photometry_df])

        n_apertures = len(apertures)
        mesh_widths, mesh_heights = np.meshgrid(aper_widths, aper_heights)

        mesh_widths = mesh_widths.flatten()
        mesh_heights = mesh_heights.flatten()
        thetas = [0] * n_apertures
        positions = [pos] * n_apertures
        zipper = zip(apertures, positions, mesh_widths,
                     mesh_heights, thetas, fluxes, errors)

        for aper, pos, aper_width, aper_height, theta, flux_, err_ in zipper:
            id_ = f'{int(time.time()*1e6):0>7}'
            self.fluxes['apertures'][id_] = aper
            self.fluxes['positions'][id_] = pos
            self.fluxes['aper_width'][id_] = aper_width
            self.fluxes['aper_height'][id_] = aper_height
            self.fluxes['thetas'][id_] = theta
            self.fluxes['fluxes'][id_] = flux_
            self.fluxes['errors'][id_] = err_

        rtime = time() - start
        msg = f'Operation took {rtime} seconds for {len(apertures)} apertures.'
        info_message(msg)

    def load_data(self, load_filename=None):
        self.fits_dict = {}
        fits_filenames = glob(f'{self.data_dir}/*{self.file_type}')
        for fname in tqdm(fits_filenames, total=len(fits_filenames)):
            key = fname.split('/')[-1].split('_')[0]
            val = fits.open(fname)
            self.fits_dict[key] = val

        if load_filename is not None:
            self.load_dict(load_filename)
        else:
            fits_filenames = glob(f'{self.data_dir}/*{self.file_type}')

            times = []
            image_stack = []
            errors_stack = []
            # fits_dict = {}
            for fname in tqdm(fits_filenames, total=len(fits_filenames)):
                key = fname.split('/')[-1].split('_')[0]
                val = fits.open(fname)
                # fits_dict[key] = val
                header = val['PRIMARY'].header
                image = val['SCI'].data
                image_stack.append(image.copy())
                errors_stack.append(val['ERR'].data)
                times.append(np.mean([header['EXPEND'], header['EXPSTART']]))

            self.image_stack = np.array(image_stack)
            self.errors_stack = np.array(errors_stack)
            self.times = np.array(times)
            # self.fits_dict = fits_dict
            self.image_shape = self.image_stack[0].shape
            self.n_images = self.image_stack.shape[0]
            self.height, self.width = self.image_shape

            self.calibration_trace_location()

        info_message(f'Found {self.n_images} {self.file_type} files')

    def calibration_trace_location(self):

        # Median Argmax
        self.median_image = np.median(self.image_stack, axis=0)
        self.mad_image = sc.mad(self.image_stack, axis=0)
        self.y_idx = np.median(self.image_stack[0].argmax(axis=0)).astype(int)

        # The median trace as the 'stellar template'
        self.median_trace = np.sum(self.median_image, axis=0)

        # Set left and right markers at halfway up the trace
        peak_trace = self.median_trace > 0.5 * self.median_trace.max()
        self.x_left = np.where(peak_trace)[0].min()
        self.x_right = np.where(peak_trace)[0].max()

    def do_fit(self, init_params=[], static_params={}):
        return
        partial_chisq = partial(chisq, times=self.times,
                                fluxes=self.fluxes,
                                errors=self.errors,
                                static_params=static_params)

        return minimize(partial_chisq, init_params)

    def batman_wrapper(self, eclipse_depth, static_params):
        return

    def chisq(self, params, static_params):
        model = batman_wrapper(params,
                               self.times,
                               static_params)

        return np.sum(((model - self.fluxes) / self.errors)**2)

    def identify_trace_direction(self):
        postargs1 = np.zeros(len(self.fits_dict))
        postargs2 = np.zeros(len(self.fits_dict))
        for k, (key, val) in enumerate(self.fits_dict.items()):
            postargs1[k] = val['PRIMARY'].header['POSTARG1']
            postargs2[k] = val['PRIMARY'].header['POSTARG2']

        postargs1_rev, postargs1_fwd = np.unique(postargs1)
        postargs2_rev, postargs2_fwd = np.unique(postargs2)

        self.idx_fwd = np.where(np.bitwise_and(postargs1 == postargs1_fwd,
                                               postargs2 == postargs2_fwd))[0]

        self.idx_rev = np.where(np.bitwise_and(postargs1 == postargs1_rev,
                                               postargs2 == postargs2_rev))[0]

    def configure_matplotlib(self):
        get_ipython().magic('config InlineBackend.figure_format = "retina"')

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.logger = logging.getLogger("theano.gof.compilelock")
        self.logger.setLevel(logging.ERROR)
        self.logger = logging.getLogger("exoplanet")
        self.logger.setLevel(logging.DEBUG)

        plt.style.use("default")
        plt.rcParams["savefig.dpi"] = 100
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["font.size"] = 16
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
        plt.rcParams["mathtext.fontset"] = "custom"

    def plot_errorbars(self, id_=None):

        id_ = list(self.fluxes['apertures'].keys())[0] if id_ is None else id_

        fluxes_ = self.fluxes['fluxes'][id_]
        fwd_fluxes_ = fluxes_[self.idx_fwd]
        rev_fluxes_ = fluxes_[self.idx_rev]

        med_flux = np.median(fluxes_)
        fwd_scatter = np.std(fwd_fluxes_ / np.median(fwd_fluxes_)) * 1e6
        rev_scatter = np.std(rev_fluxes_ / np.median(rev_fluxes_)) * 1e6

        fwd_annotate = f'Forward Scatter: {fwd_scatter:0.0f}ppm'
        rev_annotate = f'Reverse Scatter: {rev_scatter:0.0f}ppm'
        info_message(fwd_annotate)
        info_message(rev_annotate)

        fluxes_normed = fluxes_ / med_flux
        errors_normed = np.sqrt(fluxes_) / med_flux

        plt.errorbar(self.times[self.idx_fwd],
                     fluxes_normed[self.idx_fwd],
                     errors_normed[self.idx_fwd],
                     fmt='o', color='C0')

        plt.errorbar(self.times[self.idx_rev],
                     fluxes_normed[self.idx_rev],
                     errors_normed[self.idx_rev],
                     fmt='o', color='C3')

        plt.axhline(1.0, ls='--', color='C2')
        plt.title('WASP-43 HST/UVIS Observation Initial Draft Photometry')
        plt.xlabel('Time [MJD]')
        plt.ylabel('Normalized Flux')

        plt.annotate(fwd_annotate,
                     (0, 0),
                     xycoords='axes fraction',
                     xytext=(5, 5),
                     textcoords='offset points',
                     ha='left',
                     va='bottom',
                     fontsize=12,
                     color='C0',
                     weight='bold')

        plt.annotate(rev_annotate,
                     (0, 0.025),
                     xycoords='axes fraction',
                     xytext=(5, 5),
                     textcoords='offset points',
                     ha='left',
                     va='bottom',
                     fontsize=12,
                     color='C3',
                     weight='bold')

        plt.tight_layout()
        plt.show()

    def save_text_file(self, save_filename):
        info_message(f'Saving data to CSV file: {save_filename}')
        med_flux = np.median(self.fluxes)

        fluxes_normed = self.fluxes / med_flux
        errors_normed = np.sqrt(self.fluxes) / med_flux
        out_list = np.transpose([self.times, fluxes_normed, errors_normed])
        out_df = pd.DataFrame(out_list, columns=['times', 'flux', 'unc'])
        out_df.to_csv(save_filename, index=False)

    def save_dict(self, save_filename):
        info_message(f'Saving data to JobLib file: {save_filename}')
        save_dict_ = {}
        for key, val in self.__dict__.items():
            if key is not 'fits_dict' and not hasattr(val, '__call__'):
                save_dict_[key] = val

        joblib.dump(save_dict_, save_filename)

    def load_dict(self, load_filename):
        load_dict_ = joblib.load(load_filename)
        for key, val in load_dict_.items():
            if not hasattr(val, '__call__'):
                self.__dict__[key] = val
