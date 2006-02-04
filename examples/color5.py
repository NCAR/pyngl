#
#  File:
#    color5.py
#
#  Synopsis:
#    Draws a Mandelbrot set.
#
#  Category:
#    Colors
#
#  Author:
#    Fred Clare
#
#  Date of initial publication:
#    February, 2006
#
#  Description:
#    Using parameters for grid resolution and convergence tolerances,
#    this script produces a Mandelbrot set and colors it using 
#    equally-spaced hues in the HLS color space.
#
#  Effects illustrated:
#    o  Drawing filled polygons in NDC space.
#    o  Converting HLS color values to RGB.
#    o  Defining a private color table.
#
#  Output:
#    o One plot is produced showing the Mandelbrot set created.
#
import Ngl

#
#  Function for converting from user space in the
#  complex plane to NDC.
#
def user2ndc(x, y):
  return 0.4*x+0.8, 0.4*y+0.5

#
#  Function for computing the convergence of the
#  sequence for the Mandelbrot set.  
#    
#    Arguments:
#          z - the original complex value.
#        num - the maximum number of iterations allowed.
#      tolxm - a tolerance value for computed z's close to the original.
#      tollg - a tolerance value for computed z's away from the original.
#
#    Return:
#       The number of iterations it takes to satisfy the tolerance
#       criteria, or the maximum number of iterations.
# 
def convg(z, num, tolsm, tollg):
  zs = z
  z0 = z
  for i in xrange(num):
    zn = z0*z0 + zs
    if ( (abs(zn-z0) < tolsm) or (abs(zn-z0) > tollg) ):
      return i
    z0 = zn
  return num

#
#  Given the index of a cell in a grid box
#  that has lower left corner (xl,yb) and upper
#  right corner (xr,yt), and nx divisions in
#  x and ny divisions in y, return the polygon
#  bounding that cell.
#
def get_cell(i, j, xl, xr, yb, yt, nx, ny):
  delx = (xr-xl)/nx
  dely = (yt-yb)/ny
  xt = xl+i*delx
  yt = yb+j*dely
  x = [xt, xt+delx, xt+delx,      xt, xt]
  y = [yt,      yt, yt+dely, yt+dely, yt]
  return x,y

#
#  Main driver.
#
#
#  Set up corner points for the area of the
#  complex plane under consideraton.
#
xl, xr, yb, yt = -2.00, 0.50, -1.25, 1.25

#
#  Specify how many divisions in x and y the
#  area above is to have.  Increasing these
#  numbers will produce a plot with finer 
#  detail, at the expense of compute time.
#
nx, ny = 90, 90

#
#  Specify the maximum number of iterations for
#  the convergence test.  Increasing this value
#  refines the color table.  Since color values
#  are assigned to the iteration results, "niter"
#  cannot be larger than 255.
#
niter = 101

#
#  Open a workstation.
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"color5") 

#
#  Define our own color table going around 
#  all the hues in the HLS color space, starting
#  with blue.
#
for k in xrange(0,niter):
  h = 360*float(k)/float(niter)
  r,g,b = Ngl.hlsrgb(h, 50., 100.)
  Ngl.set_color(wks, k, r, g, b)

#
#  Create the set and draw the color cells.
#
poly_res = Ngl.Resources()
for j in xrange(ny):
  for i in xrange(nx):
    x, y = get_cell(i, j, xl, xr, yb, yt, nx, ny)
    iter = convg(complex(x[0],y[0]), niter, 0.001, 10000)
    poly_res.gsFillColor = iter-1   #  iter will be at least one.
    for i in xrange(len(x)):
      x[i], y[i] = user2ndc(x[i], y[i])
    Ngl.polygon_ndc(wks, x, y,poly_res)
Ngl.frame(wks)

Ngl.end()
