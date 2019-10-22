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
from scipy.interpolate import CubicSpline
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


def center_one_trace(kcol, col, fitter, stddev, y_idx, inds, buffer=10):
    model = Gaussian1D(amplitude=col.max(),
                       mean=y_idx, stddev=stddev)

    y_idx = int(y_idx)
    idx_narrow = (inds - y_idx) < buffer
    results = fitter(model, inds[idx_narrow], col[idx_narrow])
    return kcol, results, fitter


def fit_one_slopes(kimg, means, fitter, y_idx, slope_guess=2.0 / 466):
    model = Linear1D(slope=slope_guess, intercept=y_idx)

    inds = np.arange(len(means))
    inds = inds - np.median(inds)

    results = fitter(model, inds, means)

    return kimg, results, fitter


def cosmic_ray_flag_simple(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_Med = np.median(row)
        row_Std = np.std(row)
        cosmic_rays_[k] += abs(row - row_Med) > n_sig * row_Std
        image_[k][cosmic_rays_[k]] = row_Med

    return image_, cosmic_rays_


def cosmic_ray_flag_rolling(image_, n_sig=5, window=7):
    cosmic_rays_ = np.zeros(image_.shape, dtype=bool)
    for k, row in enumerate(image_):
        row_rMed = pd.Series(row).rolling(window).median()
        row_rStd = pd.Series(row).rolling(window).std()
        cosmic_rays_[k] += abs(row - row_rMed) > n_sig * row_rStd
        image_[k][cosmic_rays_[k]] = row_rMed[cosmic_rays_[k]]

    return image_, cosmic_rays_


def aper_table_2_df(aper_phots, aper_widths, aper_heights, n_images):
    info_message(f'Restructuring Aperture Photometry into DataFrames')
    if len(aper_phots) > 1:
        aper_df = aper_phots[0].to_pandas()
        for kimg in aper_phots[1:]:
            aper_df = pd.concat([aper_df, kimg.to_pandas()])
    else:
        aper_df = aper_phots.to_pandas()

    photometry_df_ = aper_df.reset_index().drop(['index', 'id'], axis=1)
    mesh_widths, mesh_heights = np.meshgrid(aper_widths, aper_heights)

    mesh_widths = mesh_widths.flatten()
    mesh_heights = mesh_heights.flatten()
    apeture_columns = [colname
                       for colname in photometry_df_.columns
                       if 'aperture_sum_' in colname]

    photometry_df = pd.DataFrame([])
    for colname in apeture_columns:
        aper_id = int(colname.replace('aperture_sum_', ''))
        aper_width_ = mesh_widths[aper_id].astype(int)
        aper_height_ = mesh_heights[aper_id].astype(int)
        newname = f'aperture_sum_{aper_width_}x{aper_height_}'

        photometry_df_[colname]
        photometry_df[newname] = photometry_df_[colname]

    photometry_df['xcenter'] = photometry_df_['xcenter']
    photometry_df['ycenter'] = photometry_df_['ycenter']

    return photometry_df


def make_mask_cosmic_rays_temporal_simple(val, kcol, krow, n_sig=5):
    val_Med = np.median(val)
    val_Std = np.std(val)
    mask = abs(val - val_Med) > n_sig * val_Std
    return kcol, krow, mask, val_Med


class HSTUVISTimeSeries(object):

    def __init__(self, planet_name='planetName', data_dir='./',
                 working_dir='./', file_type='flt.fits'):
        info_message('Initializing Instance of the `HSTUVISTimeSeries` Object')
        self.planet_name = planet_name
        self.data_dir = data_dir
        self.working_dir = working_dir
        self.file_type = file_type
        self.configure_matplotlib()

    def cosmic_ray_flag(self, image_, n_sig=5, window=7):
        return cosmic_ray_flag_simple(image_, n_sig=n_sig, window=window)

    def clean_cosmic_rays(self, n_sig=5, window=7):
        info_message('Flagging Cosmic Rays using `Temporal Simple` Technique')
        return self.clean_cosmic_rays_temporal_simple(
            n_sig=n_sig, window=window)

    def clean_cosmic_rays_temporal_rolling(self, n_sig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        for krow in tqdm(range(self.width)):
            for kcol in range(self.height):
                val = self.image_stack[:, kcol, krow]
                val_Med = pd.Series(val).rolling(window).median()
                val_Std = pd.Series(val).rolling(window).std()
                mask = abs(val - val_Med) > n_sig * val_Std
                self.cosmic_rays[:, kcol, krow] = mask
                self.image_stack[mask, kcol, krow] = val_Med[mask]

    def mp_clean_cosmic_rays_temporal_simple(self, n_sig=5, window=7):
        assert(False), 'Something is broken here'
        self.cosmic_rays = np.zeros_like(self.image_stack)

        n_pixels = self.width * self.height
        kcols, krows = np.indices((self.height, self.width))
        pixels = []
        for krow in tqdm(range(self.width)):
            for kcol in range(self.height):
                ts_now = self.image_stack[:, krow, kcol]
                pixels.append([krow, kcol, ts_now])

        # pixels = self.image_stack.reshape((n_pixels, self.n_images))

        partial_mask_cr = partial(make_mask_cosmic_rays_temporal_simple,
                                  n_sig=n_sig)

        start = time()
        pool = mp.Pool(mp.cpu_count() - 1)
        masks = pool.starmap(partial_mask_cr, zip(pixels))
        pool.close()
        pool.join()
        info_message(f'Cosmic Ray Mask Creation Took {time()-start} seconds')

        for kcol, krow, mask, val_Med in tqdm(masks):
            self.cosmic_rays[:, kcol, krow] = mask
            self.image_stack[mask, kcol, krow] = val_Med

    def clean_cosmic_rays_temporal_simple(self, n_sig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        krows, kcols = np.indices((self.height, self.width))
        start = time()
        for krow in tqdm(range(self.width)):
            for kcol in range(self.height):
                val = self.image_stack[:, kcol, krow]
                val_Med = np.median(val)
                val_Std = np.std(val)
                mask = abs(val - val_Med) > n_sig * val_Std
                self.cosmic_rays[:, kcol, krow] = mask
                self.image_stack[mask, kcol, krow] = val_Med

        info_message(f'Cosmic Ray Mask Creation Took {time()-start} seconds')

    def clean_cosmic_rays_spatial(self, n_sig=5, window=7):
        self.cosmic_rays = np.zeros_like(self.image_stack)
        for k, image_ in tqdm(enumerate(self.image_stack),
                              total=self.n_images):

            image_clean_, cosmic_rays_ = self.cosmic_ray_flag(image_,
                                                              n_sig=n_sig,
                                                              window=window)

            self.image_stack[k] = image_clean_
            self.cosmic_rays[k] = cosmic_rays_

    def center_all_traces(self, stddev=2, notit_verbose=False):
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

            # center_traces_ = [partial_center_one_trace(*entry)
            #                   for entry in zipper]

            rtime = time() - start
            info_message(f'Center computing Image {kimg} '
                         f'took {rtime:0.2f} seconds')

            for kcol, results, fitter in center_traces_:
                self.center_traces[kimg][kcol] = {}
                self.center_traces[kimg][kcol]['results'] = results
                self.center_traces[kimg][kcol]['fitter'] = fitter

    def fit_trace_slopes(self, stddev=2, notit_verbose=False):
        info_message('Fitting a slope to the Center of the Trace')
        if not hasattr(self, 'center_traces'):
            self.center_all_traces(stddev=stddev, notit_verbose=notit_verbose)

        self.gaussian_centers = np.zeros((self.n_images, self.width))
        for kimg, val0 in self.center_traces.items():
            for kcol, val1 in val0.items():
                self.gaussian_centers[kimg][kcol] = val1['results'].mean.value

        partial_fit_slp = partial(fit_one_slopes,
                                  y_idx=self.y_idx,
                                  fitter=LinearLSQFitter(),
                                  slope_guess=2.0 / 466)

        zipper = zip(np.arange(self.n_images),
                     self.gaussian_centers[:, self.x_left_idx:self.x_right_idx])

        slopInts = [partial_fit_slp(*entry) for entry in zipper]

        self.image_line_fits = {}
        for kimg, results, fitter in slopInts:
            self.image_line_fits[kimg] = {}
            self.image_line_fits[kimg]['results'] = results
            self.image_line_fits[kimg]['fitter'] = fitter

        self.trace_slopes = np.ones(self.n_images)
        self.trace_ycenters = np.ones(self.n_images)
        for kimg, val in self.image_line_fits.items():
            self.trace_slopes[kimg] = val['results'].slope.value
            self.trace_ycenters[kimg] = val['results'].intercept.value

        self.trace_angles = np.arctan(self.trace_slopes)

        if notit_verbose:
            useful = gaussian_centers.T[self.x_left_idx:self.x_right_idx]
            fig, ax = plt.subplots()
            med_trace = np.median(useful, axis=0) * 0
            ax.plot(useful - med_trace)

    def compute_sky_background(self, subpixels=5, positions=None,
                               inner_width=None, outer_width=None,
                               inner_height=None, outer_height=None,
                               thetas=None, notit_verbose=False, done_it=False):
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

        if inner_width is None:
            inner_width = 75
        if outer_width is None:
            outer_width = 150

        if inner_height is None:
            inner_height = 225
        if outer_height is None:
            outer_height = 350

        if positions is None:
            # positions = [[self.width // 2, self.y_idx]] * n_images
            xcenters_ = self.trace_xcenters
            ycenters_ = self.trace_ycenters
            positions = np.transpose([xcenters_, ycenters_])

        if thetas is None:
            thetas = self.trace_angles

        inner_width = self.trace_length + inner_width
        outer_width = self.trace_length + outer_width

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
                                              method='subpixel',
                                              subpixels=subpixels)
            outer_table = aperture_photometry(image, outer_annular,
                                              method='subpixel',
                                              subpixels=subpixels)

            inner_flux = inner_table['aperture_sum'][0]
            outer_flux = outer_table['aperture_sum'][0]
            background_area = outer_annular.area - inner_annular.area
            sky_bgs[k] = (outer_flux - inner_flux) / background_area

        self.sky_bgs = sky_bgs

    def compute_columnwise_sky_background(self, inner_height=150, edge=10):
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

        cw_sky_bgs = np.zeros((self.n_images, self.width))
        yinds, _ = np.indices(self.image_shape)

        iterator = enumerate(zip(self.image_stack, self.trace_ycenters))
        for k, (image, ycenter) in tqdm(iterator, total=self.n_images):
            mask = abs(yinds - ycenter) > inner_height
            mask = np.bitwise_and(mask, yinds > edge)
            mask = np.bitwise_and(mask, yinds < self.height - edge)
            masked_img = np.ma.array(image, mask=mask)
            cw_sky_bgs[k] = np.ma.median(masked_img, axis=0).data

        self.sky_bg_columnwise = cw_sky_bgs

    def do_phot(self, subpixels=5, positions=None,
                aper_width=None, aper_height=None,
                thetas=None, notit_verbose=False, done_it=False):
        '''
            Run photometry for a specifc set of rectangles

            Parameters
            ----------
            positions (nD-array; 2 x n_images): (xcenter, ycenter)
            aper_width (float): width of photometry aperture
            aper_height (float): height of photometry aperture
        '''

        n_images = self.n_images  # convenience for minimizing command lengths

        if positions is None:
            # positions = [[self.width // 2, self.y_idx]] * n_images
            xcenters_ = self.trace_xcenters
            ycenters_ = self.trace_ycenters
            positions = [xcenters_, ycenters_]

        if thetas is None:
            thetas = self.trace_angles

        if aper_width is None:
            aper_width = 50

        if aper_height is None:
            aper_height = 200

        aper_width = self.trace_length + aper_width
        """
        if not hasattr(self, 'fluxes'):
            self.fluxes = {}
            self.fluxes['apertures'] = {}
            self.fluxes['positions'] = {}
            self.fluxes['aper_width'] = {}
            self.fluxes['aper_height'] = {}
            self.fluxes['thetas'] = {}
            self.fluxes['fluxes'] = {}
            self.fluxes['errors'] = {}
        """
        fluxes_ = np.zeros(n_images)
        errors_ = np.zeros(n_images)

        apertures_ = []

        zipper = enumerate(zip(self.image_stack, positions, thetas))

        info_message('Creating Apertures')
        for kimg, (image, pos, theta) in tqdm(zipper, total=n_images):
            aperture = RectangularAperture(
                pos, aper_width, aper_height, theta)
            apertures_.append(aperture)

            if notit_verbose and not done_it:
                aperture = apertures_[k]
                inner_annular = self.inner_annulars[k]
                outer_annular = self.outer_annulars[k]
                plot_apertures(image, aperture, inner_annular, outer_annular)
                done_it = True

            image_table = aperture_photometry(image, aperture,
                                              method='subpixel',
                                              subpixels=subpixels)

            background = self.sky_bgs[k] * aperture.area
            fluxes_[kimg] = image_table['aperture_sum'][0] - background

        errors_ = np.sqrt(fluxes_)  # explicitly state Poisson noise limit
        """
        id_ = f'{np.random.randint(1e7):0>7}'
        self.fluxes['apertures'][id_] = apertures_
        self.fluxes['positions'][id_] = positions
        self.fluxes['aper_width'][id_] = aper_width
        self.fluxes['aper_height'][id_] = aper_height
        self.fluxes['thetas'][id_] = thetas
        self.fluxes['fluxes'][id_] = fluxes_
        self.fluxes['errors'][id_] = errors_
        """

    def do_multi_phot(self, aper_widths, aper_heights,
                      subpixels=5, positions=None, thetas=None):
        info_message('Beginning Multi-Aperture Photometry')
        # info_message(
        #     'Parameters:\n'
        #     f'AperWidths:{np.min(aper_widths)}-{np.max(aper_widths)}\n'
        #     f'AperHeights:{np.min(aper_heights)}-{np.max(aper_heights)}\n'
        #     f'SubPixels:{subpixels}\n'
        #     f'Positions:{np.median(positions)}\n'
        #     f'Thetas:{np.median(thetas)}'
        # )

        if positions is None:
            # positions = [[self.width // 2, self.y_idx]] * n_images
            xcenters_ = self.trace_xcenters
            ycenters_ = self.trace_ycenters
            positions = np.transpose([xcenters_, ycenters_])

        if thetas is None:
            thetas = self.trace_angles

        aper_widths = self.trace_length + aper_widths
        '''
        if not hasattr(self, 'fluxes'):
            self.fluxes = {}
            self.fluxes['apertures'] = {}
            self.fluxes['positions'] = {}
            self.fluxes['aper_width'] = {}
            self.fluxes['aper_height'] = {}
            self.fluxes['thetas'] = {}
            self.fluxes['fluxes'] = {}
            self.fluxes['errors'] = {}
        '''
        info_message('Creating Apertures')
        n_apertures = 0
        apertures_stack = []
        zipper_ = enumerate(zip(positions, thetas))
        for kimg, (pos, theta) in tqdm(zipper_, total=self.n_images):
            apertures_stack.append([])
            for aper_height in aper_heights:
                for aper_width in aper_widths:
                    apertures_stack[kimg].append(RectangularAperture(
                        pos, aper_width, aper_height, theta))

                    n_apertures = n_apertures + 1

        info_message('Configuing Photoutils.Aperture_Photometry')
        partial_aper_phot = partial(
            aperture_photometry, method='subpixel', subpixels=subpixels)

        zipper_ = zip(self.image_stack, self.sky_bgs)
        image_minus_sky_ = [img - sky for img, sky in zipper_]

        zipper_ = zip(image_minus_sky_, apertures_stack)

        operation = 'Aperture Photometry per Image'
        # aper_phots = [partial_aper_phot(*entry) for entry in tqdm(zipper_)]
        start = time()
        info_message(f'Computing {operation}')
        pool = mp.Pool(mp.cpu_count() - 1)
        aper_phots = pool.starmap(partial_aper_phot, zipper_)
        pool.close()
        pool.join()

        rtime = time() - start
        msg = f'{operation} took {rtime} seconds for {n_apertures} apertures.'
        info_message(msg)

        # Store raw output of all photometry to mega-list
        if hasattr(self, 'aper_phots'):
            self.aper_phots.extend(aper_phots)
        else:
            self.aper_phots = aper_phots

        if hasattr(self, 'apertures_stack'):
            self.apertures_stack.extend(apertures_stack)
        else:
            self.apertures_stack = apertures_stack

        # Convert to dataframe
        photometry_df = aper_table_2_df(
            aper_phots, np.int32(aper_widths - self.trace_length),
            np.int32(aper_heights), self.n_images)

        # Store new dataframe to object dataframe
        if not hasattr(self, 'photometry_df'):
            self.photometry_df = photometry_df
        else:
            self.photometry_df = pd.concat([self.photometry_df, photometry_df])
            self.photometry_df.reset_index().drop(['index'], axis=1)

    def load_data(self, load_filename=None):
        info_message(f'Loading Fits Files')
        self.fits_dict = {}
        fits_filenames = glob(f'{self.data_dir}/*{self.file_type}')
        for fname in tqdm(fits_filenames, total=len(fits_filenames)):
            key = fname.split('/')[-1].split('_')[0]
            val = fits.open(fname)
            self.fits_dict[key] = val

        if load_filename is not None:
            info_message(f'Loading Save Object-Dict File')
            self.load_dict(load_filename)
        else:
            info_message(f'Creating New Flux/Error/Time Attributes')
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

            times_sort = np.argsort(times)
            self.times = np.array(times)[times_sort]
            self.image_stack = np.array(image_stack)[times_sort]
            self.errors_stack = np.array(errors_stack)[times_sort]

            # self.fits_dict = fits_dict
            self.image_shape = self.image_stack[0].shape
            self.n_images = self.image_stack.shape[0]
            self.height, self.width = self.image_shape

            if not hasattr(self, 'trace_location_calibrated'):
                self.calibration_trace_location()
                self.identify_trace_direction()
                self.simple_phots()

        info_message(f'Found {self.n_images} {self.file_type} files')

    def simple_phots(self):
        self.simple_fluxes = np.zeros(self.n_images)
        for kimg, image in tqdm(enumerate(self.image_stack),
                                total=self.n_images):

            self.simple_fluxes[kimg] = np.sum(image - np.median(image))

    def calibration_trace_location(self, oversample=100):
        info_message(f'Calibration the Median Trace Location')

        # Median Argmax
        self.median_image = np.median(self.image_stack, axis=0)
        self.mad_image = sc.mad(self.image_stack, axis=0)

        # Median Trace configuration as the 'stellar template'
        self.median_trace = np.sum(self.median_image, axis=0)
        self.y_idx = np.median(self.median_image.argmax(axis=0)).astype(int)

        # Set left and right markers at halfway up the trace
        peak_trace = self.median_trace > 0.5 * self.median_trace.max()
        self.x_left_idx = np.where(peak_trace)[0].min()
        self.x_right_idx = np.where(peak_trace)[0].max()

        cs_trace = CubicSpline(np.arange(self.width), self.median_trace)
        os_xarr = np.linspace(0, self.width, self.width * oversample)
        os_trace = cs_trace(os_xarr)  # oversampled trace
        peak_trace = os_trace > 0.5 * os_trace.max()

        self.x_left = os_xarr[np.where(peak_trace)[0].min()]
        self.x_right = os_xarr[np.where(peak_trace)[0].max()]
        self.trace_length = self.x_right - self.x_left

        info_message(f'Calibration the Per Image Trace Location')
        # Trace configuration per image
        self.y_argmaxes = np.zeros(self.n_images)
        self.trace_mins = np.zeros(self.n_images)
        self.trace_maxs = np.zeros(self.n_images)
        for kimg, image in tqdm(enumerate(self.image_stack),
                                total=self.n_images):
            image_trace_ = np.sum(image, axis=0)

            yargmax_ = np.median(image_trace_.argmax(axis=0)).astype(int)
            self.y_argmaxes[kimg] = yargmax_

            cs_trace = CubicSpline(np.arange(self.width), image_trace_)
            os_trace = cs_trace(os_xarr)  # oversampled trace

            # Set left and right markers at halfway up the trace
            peak_trace_ = os_trace > 0.5 * os_trace.max()
            self.trace_mins[kimg] = os_xarr[np.where(peak_trace_)[0].min()]
            self.trace_maxs[kimg] = os_xarr[np.where(peak_trace_)[0].max()]

        self.trace_xcenters = 0.5 * (self.trace_mins + self.trace_maxs)
        self.trace_lengths = (self.trace_maxs - self.trace_mins)

        self.trace_location_calibrated = True
    """

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
    """

    def identify_trace_direction(self):
        info_message(f'Identifying Trace Direction per Image')
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
