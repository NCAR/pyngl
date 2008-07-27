#
#  File:
#    streamline3.py
#
#  Synopsis:
#    Shows how add text outside of a streamline plot.
#
#  Category:
#    Streamlines
#    Annotations on a plot
#
#  Author:
#    Mary Haley
#  
#  Date of initial publication:
#    July, 2008
#
#  Description:
#    This example draws text strings around a streamline plot
#
#  Effects illustrated:
#    o  Drawing streamlines colored by a scalar field
#    o  Annotating plots
#    o  Maximizing a plot after it is created
# 
#  Output:
#    This example produces one visualization.
#
#  Notes:
#     

#
#  Import numpy.
#
import numpy

#
#  Import PyNGL support functions.
#
import Ngl

# Create some dummy data.
def create_uv():
  N  = 30
  M  = 25
  PI = 3.14159
  uu = ((2.0 * PI)/ N) * numpy.arange(0,N,1)
  vv = ((2.0 * PI)/ M) * numpy.arange(0,M,1)
  u  = numpy.transpose(10.0 * numpy.cos(numpy.resize(uu,[M,N])))
  v  = 10.0 * numpy.cos(numpy.resize(vv,[N,M]))

  return u,v


# This function adds three subtitles to the top of a plot, left-justified,
# centered, and right-justified.

def subtitles(wks, plot, left_string, center_string, right_string, tres):
  ttres         = tres     # Copy resources
  ttres.nglDraw = False    # Make sure string is just created, not drawn.

#
# Retrieve font height of left axis string and use this to calculate
# size of subtitles.
#
  if not hasattr(ttres,"txFontHeightF"):
    font_height = Ngl.get_float(plot.base,"tiXAxisFontHeightF")
    ttres.txFontHeightF = font_height*0.8    # Slightly smaller

#
# Set some some annotation resources to describe how close text
# is to be attached to plot.
#
  amres = Ngl.Resources()
  if not hasattr(ttres,"amOrthogonalPosF"):
    amres.amOrthogonalPosF = -0.51   # Top of plot plus a little extra
                                     # to stay off the border.
  else:
    amres.amOrthogonalPosF = ttres.amOrthogonalPosF

#
# Create three strings to put at the top, using a slightly
# smaller font height than the axis titles.
#
  if left_string != "":
    txidl = Ngl.text(wks, plot, left_string, 0., 0., ttres)

    amres.amJust         = "BottomLeft"
    amres.amParallelPosF = -0.5   # Left-justified
    annoidl              = Ngl.add_annotation(plot, txidl, amres)

  if center_string != "":
    txidc = Ngl.text(wks, plot, center_string, 0., 0., ttres)

    amres.amJust         = "BottomCenter"
    amres.amParallelPosF = 0.0   # Centered
    annoidc              = Ngl.add_annotation(plot, txidc, amres)

  if right_string != "":
    txidr = Ngl.text(wks, plot, right_string, 0., 0., ttres)

    amres.amJust         = "BottomRight"
    amres.amParallelPosF = 0.5   # Right-justifed
    annoidr              = Ngl.add_annotation(plot, txidr, amres)

  return

# This function adds a string to the right axis, allowing you to have
# labels on both axes.

def right_axis(wks, plot, yaxis_string, tres):
  ttres          = tres     # Copy resources
  ttres.nglDraw  = False    # Make sure string is just created, not drawn.
  ttres.txAngleF = -90.     # Use 90 to rotate other direction.

#
# Retrieve font height of left axis string and use to calculate size of
# right axis string
#
  if not hasattr(ttres,"txFontHeightF"):
    ttres.txFontHeightF = Ngl.get_float(plot.base,"tiYAxisFontHeightF")
#
# Set up variable to hold annotation resources.
#
  amres = Ngl.Resources()

#
# Create string to put at right of plot, like a Y axis title.
#
  if yaxis_string != "":
    txid = Ngl.text(wks, plot, yaxis_string, 0., 0., ttres)

    amres.amJust           = "CenterCenter"
    amres.amParallelPosF   = 0.55   # Move towards plot.
    annoid                 = Ngl.add_annotation(plot, txid, amres)

  return

# Generate some data
u,v = create_uv()
spd =  numpy.sqrt(u**2 + v**2)

# Create plot
#
wks_type = "ps"
wks = Ngl.open_wks(wks_type,"streamline3")
Ngl.define_colormap(wks,"prcp_3")
  
res              = Ngl.Resources()          # plot mods desired
res.nglMaximize  = True
res.nglDraw      = False                    # Turn off draw and frame so
res.nglFrame     = False                    # we can attach some text.

res.nglSpreadColorStart = 5    # Control which part of colormap to use.
res.nglSpreadColorEnd   = 21

res.tiMainString      = "This is the main title"
res.tiMainFontColor   = "Navy"
res.tiMainOffsetYF    = 0.02
res.tiMainFontHeightF = 0.035
res.tiYAxisString     = "Left Y axis string"

res.stMonoLineColor   = False     # Use multiple colors for streamlines.
res.stLineThicknessF  = 2.0       # Twice as thick

res.lbOrientation            = "Horizontal"
res.pmLabelBarOrthogonalPosF = -0.02
res.pmLabelBarHeightF        = 0.1
res.pmLabelBarWidthF         = 0.6

plot = Ngl.streamline_scalar(wks,u,v,spd,res)      # Create streamline plot.

txres             = Ngl.Resources()          # Text resources desired
txres.txFontColor = "OrangeRed"

subtitles(wks,plot,"Left string","Center string","Right string",txres)

del txres.txFontColor    # Go back to foreground color (black)
txres.txFontHeightF = 0.029
right_axis(wks, plot, "Right Y axis string", txres)

Ngl.maximize_plot(wks, plot)

Ngl.draw(plot)     # Drawing the plot will draw the three subtitles attached.
Ngl.frame(wks)

Ngl.end()

