import numpy as np

# TODO: anti-aliased version, fill_coords_aa?
def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = y / img.shape[0]
            xf = x / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color

    return img

def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x-cx)*(x-cx) + (y-cy)*(y-cy) < r * r
    return fn

def point_in_rect(cx, cy, rx, ry):
    def fn(x, y):
        return abs(x-cx) < rx and abs(y - cy) < ry
    return fn

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn
