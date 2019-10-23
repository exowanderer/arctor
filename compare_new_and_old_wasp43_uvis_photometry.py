import joblib
import logging
import numpy as np
import os

from argparse import ArgumentParser
# from matplotlib import pyplot as plt

from HSTUVISTimeSeries import HSTUVISTimeSeries
from HSTUVISTimeSeries import debug_message, warning_message, info_message
# from plotting import plot_2D_stddev, plot_center_position_vs_scan_and_orbit

parser = ArgumentParser()

parser.add_argument('--planet_name', type=str, default=None)
parser.add_argument('--file_type', type=str, default='flt.fits')
parser.add_argument('--save_now', action='store_true')
parser.add_argument('--fit_model_flag', action='store_true')
parser.add_argument('--plot_verbose', action='store_true')

clargs = parser.parse_args()

clargs.plot_verbose = True
clargs.save_now = True
planet_name = 'WASP43' \
    if clargs.planet_name is None else clargs.planet_name
file_type = clargs.file_type

HOME = os.environ['HOME']
base_dir = os.path.join(HOME, 'Research', 'Planets', 'WASP43')
data_dir = os.path.join(base_dir, 'data', 'UVIS', 'MAST_2019-07-03T0738')
data_dir = os.path.join(data_dir, 'HST', 'FLTs')
working_dir = os.path.join(base_dir, 'github_analysis', 'savefiles')

wasp43_new = HSTUVISTimeSeries(
    planet_name=planet_name,
    data_dir=data_dir,
    working_dir=working_dir,
    file_type=file_type)

wasp43_stored = HSTUVISTimeSeries(
    planet_name=planet_name,
    data_dir=data_dir,
    working_dir=working_dir,
    file_type=file_type)

bestsofar = 'savefiles/WASP43_savedict_backup_221019.joblib.save'
newest = 'savefiles/WASP43_savedict_218ppm.joblib.save'

WASP43_savedict_backup_221019 = joblib.load(bestsofar)
WASP43_savedict_backup_231019 = joblib.load(newest)

for key, val in WASP43_savedict_backup_231019.items():
    if not hasattr(val, '__call__'):
        if hasattr(val, 'copy'):
            wasp43_new.__dict__[key] = val.copy()
        else:
            wasp43_new.__dict__[key] = val

for key, val in WASP43_savedict_backup_221019.items():
    if not hasattr(val, '__call__'):
        if hasattr(val, 'copy'):
            wasp43_stored.__dict__[key] = val.copy()
        else:
            wasp43_stored.__dict__[key] = val


min_aper_width = 1
max_aper_width = 100
min_aper_height = 1
max_aper_height = 300

aper_widths = np.arange(min_aper_width, max_aper_width + 2, 5)
aper_heights = np.arange(min_aper_height, max_aper_height + 2, 5)

wasp43_stored.trace_length = wasp43_stored.x_right - wasp43_stored.x_left
wasp43_stored.do_multi_phot(aper_widths, aper_heights)

# plt.figure()
# plot_2D_stddev(wasp43_stored, signal_max=235)

# plt.figure()
# plot_2D_stddev(wasp43_new, signal_max=235)
