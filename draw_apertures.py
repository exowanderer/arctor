
from astropy.visualization import simple_norm
from photutils import RectangularAperture, RectangularAnnulus

y_idx = np.median(image_stack[0].argmax(axis=0)).astype(int)
x_min = 238  # by hand
x_max = 706  # by hand

x_buffer = 50
y_width = 100
edge = 10

data = image_stack[42]

n_images, height, width = image_stack.shape

x_lower = x_min - x_buffer
x_upper = x_max + x_buffer

y_lower = y_idx - y_width
y_upper = y_idx + y_width

w_in = x_upper - x_lower
w_out = w_in + x_buffer
h_out = height - 2 * edge

pos = [width // 2, y_idx]
theta = 0

aperture = RectangularAperture(pos, w_in, 2 * y_width, theta=theta)
annular = RectangularAnnulus(
    pos, w_in + edge, w_out, h_out, theta=theta)

norm = simple_norm(image, 'sqrt', percent=99)
plt.imshow(image, norm=norm)
aperture.plot(color='white', lw=2)
annular.plot(color='red', lw=2)
