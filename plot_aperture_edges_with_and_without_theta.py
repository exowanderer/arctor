
def plot_aperture_edges_with_angle(wasp43, img_id=42):
    plt.close('all')

    y_center = wasp43.trace_ycenters[img_id]
    x_left = wasp43.x_left
    x_right = wasp43.x_right
    trace_width = x_right - x_left

    aper_tilt = RectangularAperture(
        positions[img_id], trace_width, 2, thetas[img_id])
    aper_flat = RectangularAperture(positions[img_id], trace_width, 2, 0)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(bottom=0, left=0, right=1,
                        top=0.95, hspace=0.01, wspace=0.01)

    [ax.axis('off') for axrow in axs for ax in axrow]
    [ax.imshow(image) for axrow in axs for ax in axrow]
    [ax.set_ylim(y_center - 5, y_center + 5) for axrow in axs for ax in axrow]

    aper_tilt.plot(axes=axs[0][0], color='white')
    aper_tilt.plot(axes=axs[0][1], color='white')
    aper_flat.plot(axes=axs[1][0], color='red')
    aper_flat.plot(axes=axs[1][1], color='red')

    axs[0][0].set_xlim(x_left - 10, x_left + 10)
    axs[1][0].set_xlim(x_left - 10, x_left + 10)
    axs[0][1].set_xlim(x_right - 10, x_right + 10)
    axs[1][1].set_xlim(x_right - 10, x_right + 10)

    axs[0][0].annotate('With Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='white')

    axs[0][1].annotate('With Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='white')

    axs[1][0].annotate('Without Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='red')

    axs[1][1].annotate('Without Calculated Tilt',
                       (0, 0),
                       xycoords='axes fraction',
                       xytext=(5, 5),
                       textcoords='offset points',
                       ha='left',
                       va='bottom',
                       fontsize=20,
                       color='red')

    fig.suptitle('Example Aperture With and Without Rotation', fontsize=20)
