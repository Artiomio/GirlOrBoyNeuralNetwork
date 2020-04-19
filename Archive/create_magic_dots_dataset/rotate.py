from math import sin, cos
def rotate(coords, origin, angle):
    """ Rotates given point around given origin
    """
    x, y = coords
    xc, yc = origin

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    x_vector = x - xc
    y_vector = y - yc

    x_new = x_vector * cos_angle - y_vector * sin_angle + xc
    y_new = x_vector * sin_angle + y_vector * cos_angle + yc
    return (x_new, y_new)
