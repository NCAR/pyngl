load "$NCARG_ROOT/lib/ncarg/nclex/gsun/gsn_code.ncl"

begin
  colormaps= (/ \
                "3gauss",  "3saw", "BkBlAqGrYeOrReViWh200",  \
                "BkBlAqGrYeOrReViWh200", "BlAqGrYeOrRe", \
                "BlAqGrYeOrRe",  "BlAqGrYeOrReVi200", \
                "BlAqGrYeOrReVi200",  "BlGrYeOrReVi200", \
                "BlGrYeOrReVi200",  "BlRe",  "BlWhRe", \
                "BlWhRe",  "BlueRed",  "BlueRedGray", \
                "GreenYellow",  "ViBlGrWhYeOrRe", \
                "ViBlGrWhYeOrRe",  "WhBlGrYeRe", \
                "WhBlGrYeRe",  "WhBlReWh", "WhViBlGrYeOrRe", \
                "WhViBlGrYeOrRe", "WhViBlGrYeOrReWh", \
                "WhViBlGrYeOrReWh", "amwg", "cosam", "cosam12", \
                "cyclic", "default", "detail", "example", \
                "extrema", "gscyclic", "gsltod", "gsdtol", \
                "gui_default", "helix",  "helix1", \
                "hlu_default", "hotres","ncview_default", \
                "nrl_sirkes", "psgcap", "rainbow+gray", \
                "rainbow+white+gray","rainbow+white", \
                "rainbow",  "so4_21", "so4_23",  "tbrAvg1", \
                "tbrStd1", "tbrStd1",  "tbrVar1",  "tbrVar1", \
                "tbr_240-300",  "tbr_240-300", "tbr_stdev_0-30", \
                "tbr_stdev_0-30", "tbr_var_0-500", \
                "tbr_var_0-500", "temp1", "testcmap", "thelix", \
                "uniform", "wgne15", "wh-bl-gr-ye-re", "wxpEnIR"/)

  ncolormaps = dimsizes(colormaps)

;
; The type is being mixed here (double & float) for testing
; purposes.
;

  xmark1 = (/-999.,0.32d,0.33,0.34,0.35,0.36,-999.,0.37/)
  ymark1 = (/   0.,0.15,0.15,0.15,0.15,0.15, -99.,0.15/)
  xmark2 = (/0.36,0.35,0.34,0./)
  ymark2 = (/0.16d,0.17d,0.18d,-999./)
  xmark3 = (/0.33,0.32,0.31/)
  ymark3 = (/0.17,0.16,0.15/)
  xline1 = (/-9.,0.01, 0.33, -9., 0.50, -1., -9., 0.60, -9., 0.80, 0.99/)
  yline1 = (/-8.,  0.05, 0.05, -8., 0.05, -8.,  0., 0.05, -8., 0.05, 0.05/)

  xline2  = (/6.,15.,15.,0.,15.,0.,15.,15.,6.,6.,0.,6.,0.,6.,6./)
  yline2  = (/271.5,271.5,272.3,-999.0,272.4,-999.0,272.5, \
              273.1,273.1,271.9,-999.0,271.8,-999.0,271.7,271.5/)

  xgon  = (/-999., 0.80, 0.90, 0.90, -999., 0.80, 0.80/)
  ygon  = (/   0., 0.05, 0.05, 0.10, -999., 0.10, 0.05/)
  ygon2 = (/   0.,  0.75, 0.75, 0.80,    0., 0.80, 0.75/)

  mapxgon = (/-120, -65, -65, -120, -120/)
  mapygon = (/  60,  60,  59,   59 ,  60/)

  mapxline = (/-999, -110,   -75, -999, -115,  -70, \
               -999, -120,   -65, -999, -125,  -60, \
               -999, -130,   -55, -999, -135,  -50/)

  mapyline = (/-999,   50,    50, -999,   45,   45, \
               -999,   40,    40, -999,   35,   35, \
               -999,   30,    30, -999,   25,   25/)

  xmark1@_FillValue = -999.
  ymark1@_FillValue = -99.
  ymark2@_FillValue = -999.
  xline1@_FillValue = -9.
  yline1@_FillValue = -8.
  yline2@_FillValue = -999.
  xgon@_FillValue   = -999
  ygon@_FillValue   = -999.
  mapxline@_FillValue = -999
  mapyline@_FillValue = -999

  latgon = (/  20,  60,  60,  20,  20/)
  longon = (/-125,-125, -65, -65, -125/)

  cncolors = (/2,16,30,44,58,52,86,100,114,128,142,156,170/)
  mpcolors = (/0, -1, 238, -1/)
  pttrns   = (/0,1,2,3,4,5,6,7,8,9,10,11,12/)

;
; Open the netCDF files for contour and vector data.
;
  fT  = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/meccatemp.cdf", "r")
  fU  = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/Ustorm.cdf", "r")
  fV  = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/Vstorm.cdf", "r")
  fP  = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/Pstorm.cdf", "r")
  fT2 = addfile("$NCARG_ROOT/lib/ncarg/data/cdf/Tstorm.cdf", "r")

  t  = fT->t(0,:,:)
  t2 = fT2->t(0,:,:)
  t2 = (t2-273.15)*(9./5.) + 32.
  u  = fU->u(0,:,:)
  v  = fV->v(0,:,:)
  p  = (fP->p(0,:,:)) * 0.01

;
; Initialize which plots to draw.
;
  do_colormaps         = False
  do_contour           = True
  do_y                 = True
  do_xy_multi          = True
  do_xy_single         = True
  do_vector            = True
  do_streamline        = True
  do_map               = True
  do_contour_map       = True
  do_contour_map2      = True
  do_vector_scalar     = True
  do_vector_map        = True
  do_streamline_map    = True
  do_vector_scalar_map = True

;
; Initialize color map for later.
;
   cmap = (/(/ 1.00, 1.00, 1.00/), \
            (/ 0.00, 0.00, 0.00/), \
            (/ .560, .500, .700/), \
            (/ .300, .300, .700/), \
            (/ .100, .100, .700/), \
            (/ .000, .100, .700/), \
            (/ .000, .300, .700/), \
            (/ .000, .500, .500/), \
            (/ .000, .700, .100/), \
            (/ .060, .680, .000/), \
            (/ .550, .550, .000/), \
            (/ .570, .420, .000/), \
            (/ .700, .285, .000/), \
            (/ .700, .180, .000/), \
            (/ .870, .050, .000/), \
            (/ 1.00, .000, .000/), \
            (/ .800, .800, .800/)/)

;
; Set up workstation resource list.
;
  wks = gsn_open_wks("ps","driver2")

;
; Set color map resource and open workstation.
;

  if(do_colormaps) then
    do i = 0, ncolormaps-1
      setvalues wks
        "wkColorMap" : colormaps(i)
      end setvalues
      gsn_draw_colormap(wks)
    end do
  end if

;
; Initialize colormap back to "rainbow+gray".
;
  setvalues wks
    "wkColorMap" : "rainbow+gray"
  end setvalues

;
; gsn_contour section
;

  if(do_contour) then
    cnres = True
    txres = True
    gsres = True
;
; Set some contour resources.
;
    cnres@gsnMaximize          = True
    cnres@sfXArray             = t&lon
    cnres@sfYArray             = t&lat
    cnres@cnFillOn             = True
    cnres@cnFillColors         = cncolors
    cnres@cnLineLabelsOn       = False
    cnres@lbPerimOn            = False
    cnres@pmLabelBarDisplayMode= "ALWAYS"

;
; Set some text resources and draw a text string.
;
    txres@txFontHeightF = 0.02
    txres@txFont        = 21
    txres@txJust        = "CenterLeft"
    txres@txFuncCode    = "~"

    xf = 0.01
    yf = 0.05
    gsn_text_ndc(wks,"gsn_text_ndc: bottom feeder",xf,yf,txres)

;
; Draw a polygon before we draw the plot.
;
    gsres                 = True
    gsres@gsFillColor     = "SlateBlue"
    gsres@gsEdgesOn       = True
    gsres@gsEdgeThicknessF= 3.0
    gsres@gsEdgeColor     = "Salmon"
    gsn_polygon_ndc(wks, xgon, ygon, gsres)

;
; Create and draw contour plot, and advance frame.
;
    contour = gsn_contour(wks, t, cnres)

    delete(cnres)
    delete(txres)
    delete(gsres)
  end if

;
; gsn_y section
;
; Index values are used for the X axis.
;

  if(do_y) then
    txres               = True
    txres@txFuncCode    = "~"
    txres@txFontHeightF = 0.03
    txres@txJust        = "TopLeft"
    txres@txDirection   = "Down"

    xyres               = True
    xyres@gsnMaximize   = True
    xyres@gsnFrame      = False
    xyres@gsnDraw       = False

    xy = gsn_y(wks, t(0,:), xyres)

    xf = 0.3
    yf = 0.8
    gsn_text_ndc(wks,"gsn_text_ndc",xf,yf,txres)

    xf = 0.35
    gsn_text_ndc(wks,"Down",xf,yf,txres)

    draw(xy)

    gsres                   = True
    gsres@gsLineColor       = "red"
    gsres@gsLineDashPattern = 11
    gsn_polyline(wks, xy, xline2, yline2, gsres)

    frame(wks)
    
    delete(xyres)
    delete(txres)
    delete(gsres)
  end if

;
; gsn_xy section
;

  if(do_xy_multi) then
    xyres = True
    txres = True
    gsres = True

    xyres@trXMinF         = -180
    xyres@trXMaxF         =  180
    xyres@tiMainFont      = "helvetica-bold"
    xyres@tiMainFontColor = "red"
    xyres@tiMainString    = "This is a boring red title"
;
; Resources for multiple lines.
;
    xyres@xyMonoLineColor = False
    xyres@xyLineColors    = cncolors
    xyres@xyDashPatterns  = pttrns
    xyres@gsnFrame = False
    xyres@gsnDraw  = False
    xyres@gsnMaximize   = True

;
; Create and draw XY plot, and advance frame. In this case, the X data
;
; Plot only the first 10 lines.
;

    xy = gsn_xy(wks, t&lon, t(0:9,:), xyres)
;
; Set some text resources and draw a text string.
;
    txres@txAngleF     =   90.
    txres@txFontHeightF= 0.03
    txres@txFuncCode   = "~"

    ixf = -130
    iyf =  268
    text1 = gsn_add_text(wks, xy, "~F26~gsn_add_text:~C~sideways",  \
                         ixf, iyf, txres)

    draw(xy)

    txres@txAngleF=     270.
    txres@txFontColor=  "green"
    ixf = -80
    gsn_text(wks, xy, "~F26~gsn_text:~C~green, sideways", ixf, iyf, txres)

    frame(wks)
    delete(xyres)
    delete(txres)
  end if

;
; Plot just a single line.
;
  if(do_xy_single) then
    xyres = True
    gsres = True

    xyres@xyLineColor      = "green"
    xyres@xyLineThicknessF = 3.0
    xyres@gsnFrame         = False
    xyres@gsnMaximize      = True

    xy = gsn_xy(wks, t&lat, t(0,:), xyres)

    gsres@gsMarkerColor= "Red"
    gsres@gsMarkerIndex= 17
    gsres@gsMarkerSizeF= 0.02

    xm = -70.
    ym = 271.5
    gsn_polymarker(wks, xy, xm, ym, gsres)

    xm = -50.
    gsres@gsMarkerSizeF = 0.03
    gsn_polymarker(wks, xy, xm, ym, gsres)

    xm = -30.
    gsres@gsMarkerSizeF = 0.04
    gsn_polymarker(wks, xy, xm, ym, gsres)

    xm = -10.
    gsres@gsMarkerSizeF= 0.05
    gsn_polymarker(wks, xy, xm, ym, gsres)

    xm =  10.
    gsres@gsMarkerSizeF= 0.06
    gsn_polymarker(wks, xy, xm, ym, gsres)

    frame(wks)
 
    delete(xyres) 
    delete(gsres)
 end if

;
; gsn_streamline section
;

  if(do_streamline) then
    stres = True

;
; Create and draw streamline plot, and advance frame.
;
    stres@vfXArray    = u&lon
    stres@vfYArray    = u&lat
    stres@gsnMaximize = True

    streamline = gsn_streamline(wks, u, v, stres)
    delete(stres)
  end if

;
; gsn_vector section
;

  if(do_vector) then
    vcres = True
;
; Set some vector resources.
;
    vcres@vfXArray             = u&lon
    vcres@vfYArray             = u&lat
    vcres@vcRefAnnoOrthogonalPosF= -0.2
    vcres@vpXF=      0.10
    vcres@vpYF=      0.95
    vcres@vpWidthF=  0.85
    vcres@vpHeightF= 0.85
    vcres@gsnMaximize = False
;
; Create and draw vector plot, and advance frame.
;
    vector = gsn_vector(wks, u, v, vcres)

    delete(vcres)
  end if

;
; gsn_map section
;

  if(do_map) then
    mpres = True
;
; Set some map resources.
;
    mpres@mpGridAndLimbOn    = False
    mpres@mpPerimOn          = 1
    mpres@tiMainString       = "CylindricalEquidistant"
    mpres@gsnMaximize        = True

;
; Create and draw map plot, and advance frame.
;
    map = gsn_map(wks, "CylindricalEquidistant", mpres)

    delete(mpres)
  end if

;
; gsn_contour_map section
;

  if(do_contour_map) then
    mpres = True

;
; Set up some resources.
;
    mpres@sfXArray              = t&lon
    mpres@sfYArray              = t&lat
    mpres@cnFillOn              = True
    mpres@cnFillColors          = cncolors
    mpres@cnLinesOn             = False
    mpres@cnLineLabelsOn        = False
    mpres@cnInfoLabelOn         = False
    mpres@pmLabelBarDisplayMode = "ALWAYS"
    mpres@pmTickMarkDisplayMode = "ALWAYS"
    mpres@lbOrientation         = "Horizontal"
    mpres@lbPerimOn             = False
    mpres@pmLabelBarSide        = "Bottom"

    mpres@mpFillOn              = True
    mpres@mpFillColors          = mpcolors
    mpres@mpFillDrawOrder       = "PostDraw"
    mpres@mpGridAndLimbOn       = False
    mpres@mpGeophysicalLineColor="black"
    mpres@mpPerimOn             = 1

    mpres@gsnSpreadColors      = True
    mpres@gsnSpreadColorStart  = 3
    mpres@gsnSpreadColorEnd    = -2
    mpres@gsnMaximize          = True

    cntrmap = gsn_contour_map(wks, t, mpres)

    delete(mpres)
  end if

;
; gsn_contour_map section
;

  if(do_contour_map2) then
    mpres = True
    txres = True
;
; Ser color map back to the default.
;
    setvalues wks
      "wkColorMap"        : "hlu_default"
      "wkForegroundColor" : "black"
      "wkBackgroundColor" : "white"
    end setvalues

;
; Set up some resources.
;
    mpres@cnFillOn              = True
    mpres@cnLineLabelsOn        = False
    mpres@cnInfoLabelOn         = False
    mpres@pmLabelBarDisplayMode = "Always"
    mpres@lbPerimOn             = False

    mpres@sfXCStartV = -140.0
    mpres@sfXCEndV   =  -52.5
    mpres@sfYCStartV =   20.0
    mpres@sfYCEndV   =   60.0

    mpres@mpProjection = "LambertEqualArea"
    mpres@mpCenterLonF = -96.25
    mpres@mpCenterLatF =  40.0

    mpres@mpLimitMode = "LatLon"
    mpres@mpMinLonF   = -140.0
    mpres@mpMaxLonF   =  -52.5
    mpres@mpMinLatF   =   20.0
    mpres@mpMaxLatF   =   60.0
    mpres@mpPerimOn  = True

    mpres@tiMainString = ":F26:January 1996 storm"
    mpres@pmTickMarkDisplayMode = "ALWAYS"

    mpres@vpXF      = 0.09
    mpres@vpYF      = 0.80
    mpres@vpWidthF  = 0.65
    mpres@vpHeightF = 0.50

    mpres@gsnMaximize = False
    mpres@gsnScale    = True
    mpres@gsnFrame    = False

    cntrmap = gsn_contour_map(wks, p, mpres)
;
; Set some text resources.
;

    txres@txFontHeightF = 0.02
    txres@txFontColor   = 4
    xf = 0.90
    yf = 0.35
    gsn_text_ndc(wks,":F25:Pressure (mb)",xf,yf,txres)
    
    frame(wks)
  
    delete(mpres)
    delete(txres)
  end if

;
; Define a new color map.
;
  setvalues wks
    "wkColorMap" : cmap
  end setvalues

;
; gsn_vector_map section
;

 if(do_vector_scalar) then
   vcres = True
   gsres = True
;
; First set up some resources.
;
    vcres@vfXArray               = u&lon
    vcres@vfYArray               = u&lat
    vcres@sfXArray               = t2&lon
    vcres@sfYArray               = t2&lat

    vcres@pmLabelBarDisplayMode  = "Always"
    vcres@pmLabelBarSide         = "Bottom"
    vcres@lbOrientation          = "Horizontal"
    vcres@lbPerimOn              = False
    vcres@lbTitleFont            = 25
    vcres@lbTitleString          = "TEMPERATURE (:S:o:N:F)"
    vcres@tiMainFontHeightF      = 0.03
    vcres@tiMainString           = ":F25:Wind velocity vectors"
    
    vcres@vcFillArrowEdgeColor    = 1
    vcres@vcFillArrowsOn          = True
    vcres@vcMinFracLengthF        = 0.33
    vcres@vcMinMagnitudeF         = 0.001
    vcres@vcMonoFillArrowFillColor= False
    vcres@vcMonoLineArrowColor    = False
    vcres@vcRefLengthF            = 0.045
    vcres@vcRefMagnitudeF         = 20.0

;
; Create and draw plot; don't advance frame.
;
    vcres@gsnSpreadColors     = True
    vcres@gsnSpreadColorStart = 3
    vcres@gsnSpreadColorEnd   = -2
    vcres@gsnFrame            = False
    vcres@gsnMaximize         = True
    vctrscalar = gsn_vector_scalar(wks, u, v, t2, vcres)

;
; Draw some polymarkers on canvas.
;
 
    gsres@gsMarkerIndex= 16
    gsres@gsMarkerSizeF= 10.5
    gsres@gsMarkerColor= "red"
    gsn_polymarker_ndc(wks, xmark1, ymark1, gsres)

    gsres@gsMarkerColor= "green"
    gsn_polymarker_ndc(wks, xmark2, ymark2, gsres)

    gsres@gsMarkerColor= "blue"
    gsn_polymarker_ndc(wks, xmark3, ymark3, gsres)

    frame(wks)

    delete(vcres)
    delete(gsres)
  end if

;
; gsn_vector_map section
;

  if(do_vector_map) then
    mpres = True
    gsres = True
    txres = True

;
; Set some vector resources.
;
    mpres@vfXArray                = u&lon
    mpres@vfYArray                = u&lat
    mpres@vcFillArrowEdgeColor    = 1
    mpres@vcFillArrowsOn          = True
    mpres@vcMinFracLengthF        = 0.33
    mpres@vcMinMagnitudeF         = 0.001
    mpres@vcMonoFillArrowFillColor= False
    mpres@vcMonoLineArrowColor    = False
    mpres@vcRefLengthF            = 0.045
    mpres@vcRefMagnitudeF         = 20.0
    mpres@vcRefAnnoOrthogonalPosF= -0.1
    mpres@pmLabelBarDisplayMode   = "Always"
    mpres@pmLabelBarSide          = "Bottom"
    mpres@lbOrientation           = "Horizontal"
    mpres@lbPerimOn               = False
    mpres@lbLabelFont             = 25

;
; Set some map resources.
;
    mpres@mpProjection           = "Mercator"
    mpres@mpLimitMode            = "LatLon"
    mpres@mpMaxLatF              =  65.0
    mpres@mpMaxLonF              = -58.
    mpres@mpMinLatF              =  18.0
    mpres@mpMinLonF              = -128.
    mpres@mpCenterLatF           =   40.0
    mpres@mpCenterLonF           = -100.0
    mpres@mpFillOn               = True
    mpres@mpInlandWaterFillColor = -1
    mpres@mpLandFillColor        = "LightGray"
    mpres@mpOceanFillColor       = -1
    mpres@mpGridLineDashPattern  = 2
    mpres@mpGridMaskMode         = "MaskNotOcean"
    mpres@mpPerimOn              = True
    mpres@mpOutlineBoundarySets  = "GeophysicalAndUSStates"
    mpres@gsnFrame               = False
    mpres@gsnMaximize            = True

    vctrmap = gsn_vector_map(wks, u, v, mpres)
  
;
; Draw a polyline before we draw the plot.
;
    gsres@gsLineColor     = "red"
    gsres@gsLineThicknessF= 2.5
    gsn_polyline_ndc(wks, xline1, yline1, gsres)

;
; Set some text resources and draw a text string.
;
    txres@txFontHeightF= 0.03
    txres@txFont       = 22
    txres@txFuncCode   = "~"

    xf = 0.5
    yf = 0.16
    gsn_text_ndc(wks,"gsn_text_ndc: I'm a big labelbar", xf,yf,txres)
    frame(wks)

    delete(mpres)
    delete(gsres)
    delete(txres)
  end if

;
; gsn_streamline_map section
;

  if(do_streamline_map) then
    mpres = True
    gsres = True
    txres = True
;
; Set some resources for streamline over map plot.
;
    mpres@mpProjection           = "LambertConformal"
    mpres@mpLambertParallel1F    =   0.001
    mpres@mpLambertParallel2F    =  89.999
    mpres@mpLambertMeridianF     = -93.0
    mpres@mpLimitMode            = "LatLon"
    mpres@mpMaxLatF              =  65.0
    mpres@mpMaxLonF              = -58.
    mpres@mpMinLatF              =  18.0
    mpres@mpMinLonF              = -128.
    mpres@mpPerimOn              = True

;
; Set some streamline resources. 
;
    mpres@vfXArray                = u&lon
    mpres@vfYArray                = u&lat
    mpres@stLineColor             = "green"
    mpres@stLineThicknessF        = 2.0
    mpres@tiMainString            = "Green Streams"
    mpres@tiMainFont              = 25

    mpres@gsnMaximize             = True
    mpres@gsnFrame                = False
    mpres@gsnDraw                 = False

    strmlnmap = gsn_streamline_map(wks, u, v, mpres)

;
; Draw the polylines first, then draw the streamline
; plot. The rest of the primitives will be drawn on top.
;
    gsres@gsLineColor      = "Brown"
    gsres@gsLineThicknessF = 3.0

    gsn_polyline(wks, strmlnmap, mapxline, mapyline, gsres)

    draw(strmlnmap)

;
; Set some text resources and draw a text string.
;
    txres@txFontHeightF        = 0.02
    txres@txFuncCode           = "~"
    txres@txFont               = "helvetica-bold"
    txres@txFontColor          = "Blue"
    txres@txPerimOn            = True
    txres@txBackgroundFillColor= "LightGray"
    
    ixf = -93
    iyf =  65
    gsn_text(wks, strmlnmap,"gsn_text: lat=65,lon=-93", ixf, iyf, txres)
;
; Draw a polygon, markers, and line.
;
    delete(gsres)
    gsres            = True
    gsres@gsFillColor= "turquoise"
    gsn_polygon_ndc(wks, xgon, ygon2, gsres)

    gsres@gsMarkerColor = "red"
    gsres@gsMarkerSizeF = 8.2
    gsres@gsMarkerSizeF = 8.2
    gsres@gsMarkerIndex =  10
    gsn_polymarker_ndc(wks, xgon, ygon2, gsres)

    gsres@gsLineThicknessF = 1.0
    gsres@gsLineColor      = "Blue"
    gsn_polyline_ndc(wks, xgon, ygon2, gsres)

    gsres@gsFillColor= "Red"
    gsn_polygon(wks, strmlnmap, mapxgon, mapygon, gsres)

    frame(wks)

    delete(mpres)
    delete(txres)
    delete(gsres)
  end if

;
; gsn_vector_scalar_map section
;

  if(do_vector_scalar_map) then
    mpres = True
    gsres = True
    txres = True
;
; Set some map resources.
;
    mpres@mpProjection           = "Mercator"
    mpres@mpLimitMode            = "LatLon"
    mpres@mpMaxLatF              =  60.0
    mpres@mpMaxLonF              = -62.
    mpres@mpMinLatF              =  18.0
    mpres@mpMinLonF              = -128.
    mpres@mpCenterLatF           =   40.0
    mpres@mpCenterLonF           = -100.0
    mpres@mpFillOn               = True
    mpres@mpInlandWaterFillColor = -1
    mpres@mpLandFillColor        = "LightGray"
    mpres@mpOceanFillColor       = -1
    mpres@mpGridLineDashPattern  = 2
    mpres@mpGridMaskMode         = "MaskNotOcean"
    mpres@mpOutlineOn            = False
    mpres@mpPerimOn              = True

;
; Set some vector resources.
;

    mpres@pmLabelBarDisplayMode  = "Always"
    mpres@pmLabelBarSide         = "Bottom"
    mpres@lbOrientation          = "Horizontal"
    mpres@lbPerimOn              = False
    mpres@lbTitleFont            = 25
    mpres@lbTitleString          = "TEMPERATURE (:S:o:N:F)"
    mpres@tiMainFontHeightF      = 0.03
    mpres@tiMainString           = ":F25:Wind velocity vectors"

    mpres@vfXArray                = u&lon
    mpres@vfYArray                = u&lat
    mpres@vcFillArrowEdgeColor    = 1
    mpres@vcFillArrowsOn          = True
    mpres@vcMinFracLengthF        = 0.33
    mpres@vcMinMagnitudeF         = 0.001
    mpres@vcMonoFillArrowFillColor= False
    mpres@vcMonoLineArrowColor    = False
    mpres@vcRefLengthF            = 0.045
    mpres@vcRefMagnitudeF         = 20.0
    mpres@vcGlyphStyle            = "CurlyVector"

    mpres@gsnFrame                = False
    mpres@gsnDraw                 = False
    mpres@gsnMaximize             = True

    vctrmap = gsn_vector_scalar_map(wks, u, v, t2, mpres)

;
; Attach a polygon and outline it using a polyline. Note: it is possible
; to outline a polygon by setting the polygon resource "gsEdgesOn" to 
; True.  We're doing it using a polyline for test purposes.
;
    gsres@gsFillIndex= 17
    i1 = gsn_add_polygon(wks, vctrmap, longon, latgon, gsres)
    gsres@gsLineThicknessF= 2.0
    i2 = gsn_add_polyline(wks, vctrmap, longon, latgon, gsres)

;
; Mark the four corners of the polygon with polymarkers.
;
    gsres@gsMarkerIndex= 16
    gsres@gsMarkerSizeF= 10.5
    i3 = gsn_add_polymarker(wks, vctrmap, longon, latgon, gsres)

;
; Label two corners of the polygon with text.
;
    ixf = -125
    yf  =   60
    txres@txFontHeightF= 0.02
    txres@txFont       = "helvetica"
    txres@txJust       = "BottomRight"

    text1 = gsn_add_text(wks, vctrmap, "lat=  60:C:lon=-125", ixf, yf, txres)

    ixf = -65
    yf  =  60
    txres@txJust= "BottomLeft"
    text2 = gsn_add_text(wks, vctrmap,"lat= 60:C:lon=-65", ixf, yf, txres)
;
; Once you've "attached" all the primitives and text you want, go
; ahead and draw the map.  You will see all the primitives and text
; drawn too, since they are attached to the map.
;
    draw(vctrmap)

;
; Label the other two corners of the polygon with text.
;
    
    txres@txJust= "TopRight"
;
; Don't use -125 here, because this put the label on the far left edge of
; the plot, and it will be cut if you go to PS/PDF output with maximize on.
;    ixf = -125;
;
    ixf = -120
    yf  =   20
    gsn_text(wks, vctrmap,"lat=  20:C:lon=-125", ixf, yf, txres)

    ixf = -65
    yf  =  20
    txres@txJust= "TopLeft"
    gsn_text(wks, vctrmap,"lat= 20:C:lon=-65", ixf, yf, txres)

    frame(wks)

;
; Change viewport coordinates to show how some text gets resized, and
; some doesn't.
; 
    setvalues vctrmap
      "vpXF"      :   0.3
      "vpYF"      :   0.7
      "vpHeightF" :   0.4
      "vpWidthF"  :   0.4
    end setvalues

    draw(vctrmap)
    frame(wks)
  end if

;
; NhlDestroy destroys the given id and all of its children.
;

  destroy(wks)
end

