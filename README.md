# Arctor
A object-oriented pipeline for the HST UVIS instrument to extract photometry from scanning mode HST observations and other arc-like observations that require a rectangular aperture. 

This package is designed to work with transit & eclipse observations for the HST-UVIS photometric scanning mode. The code uses high precision, optimal photometry to track the arc of starlight caused by scanning HST during an observations of an exoplanet host star.  

It tracks the xcenter, ycenter, y-fwhm, and x-trace length; computes the sky background (using both median aperture and column-wise estimations), and cleans cosmic rays. Using the collection of ycenters along the trace, `Arctor` measures the position and rotation of the arc in each image through the transit, eclipse, or phase curve.  Using this information, it computes the flux inside a rotated rectangular aperture [for photometry and median sky background estimation].
