load "$NCARG_ROOT/lib/ncarg/nclex/gsun/gsn_code.ncl"

function create_title(parray, parray_copy, nplots, panel_dims, \
                      extra_string:string, row_spec:logical)
begin
  title = new(1,string)
  ndims = dimsizes(panel_dims)

  if(row_spec) then
    title = ":F25:RowSpec = " + panel_dims(0)
    do i=1,ndims-1
      title = title + ", " + panel_dims(i)
    end do
  else
    title = ":F25:" + panel_dims(0) + " rows/" + panel_dims(1) + " columns"
  end if

  first_time = True
  do i = 0,nplots-1
    if(ismissing(parray(i))) then
      if(first_time) then
        title = title + ":C:plots " + i
        first_time = False
      else
        title = title + ", " + i
      end if
      parray(i) = parray_copy(i)
    end if
  end do
  if(.not.first_time) then
    title = title + " are missing"
  end if
  if(extra_string.ne."") then
    title = title + ":C:" + extra_string
  end if
  return(title)
end

begin
   cmap = (/(/1.00, 1.00, 1.00/), (/0.00, 0.00, 0.00/), \
            (/.560, .500, .700/), (/.300, .300, .700/), \
            (/.100, .100, .700/), (/.000, .100, .700/), \
            (/.000, .300, .700/), (/.000, .500, .500/), \
            (/.000, .700, .100/), (/.060, .680, .000/), \
            (/.550, .550, .000/), (/.570, .420, .000/), \
            (/.700, .285, .000/), (/.700, .180, .000/), \
            (/.870, .050, .000/), (/1.00, .000, .000/), \
            (/.800, .800, .800/)/)


  cdfpath = ncargpath("data") + "/cdf/"
  a = addfile(cdfpath + "Ustorm.cdf","r")
  b = addfile(cdfpath + "Vstorm.cdf","r")
  c = addfile(cdfpath + "Tstorm.cdf","r")

  u = a->u
  v = b->v
  t = c->t
  t = (t-273.15)*(9./5.) + 32.

  wks = gsn_open_wks("ps","panel2")

;
; Set some contour/vector common resources.
;

  res = True
  res@gsnDraw                = False
  res@gsnFrame               = False
  res@gsnSpreadColors        = True

  if(isatt(res,"gsnSpreadColors").and.res@gsnSpreadColors) then
    gsn_define_colormap(wks,"rainbow")
  else
    gsn_define_colormap(wks,cmap)
  end if

;
; Set some vector resources.
;
  vcres                          = res
  vcres@vfYArray                 = t&lat
  vcres@vfXArray                 = t&lon
  vcres@vcRefLengthF             =  0.045
  vcres@vcRefMagnitudeF          = 20.0
  vcres@vcMinMagnitudeF          = 0.001
  vcres@vcFillArrowsOn           = "True"
  vcres@vcMonoFillArrowFillColor = "False"
  vcres@vcMinFracLengthF         = 0.33
  vcres@pmLabelBarDisplayMode    = "Never"
  vcres@vcLevelSelectionMode     = "ManualLevels"
  vcres@vcLevelSpacingF          = 2.0
  vcres@vcMinLevelValF           = 0.0
  vcres@vcMaxLevelValF           = 20.0

; 
; Set some contour resources.
;
  cnres = res
  cnres@sfYArray               = t&lat
  cnres@sfXArray               = t&lon

  cnres@pmLabelBarDisplayMode  = "Always"
  cnres@pmLabelBarSide         = "Bottom"
  cnres@lbPerimOn              = "False"
  cnres@lbTitleFont            = 25
  cnres@lbTitleString          = "temperature (:S:o:N:F)"
  cnres@pmLabelBarDisplayMode  = "Never"
  cnres@tiMainFontHeightF      = 0.03
  cnres@tiMainString           = ":F25:Wind velocity vectors"
  cnres@cnFillOn               = True
  cnres@cnLevelSelectionMode   = "ManualLevels"
  cnres@cnMinLevelValF         = -20.
  cnres@cnMaxLevelValF         =  80.
  cnres@cnLevelSpacingF        =  10.

  ntime = dimsizes(t(:,0,0))

  carray      = new(ntime,graphic)
  carray_copy = new(ntime,graphic)
  varray      = new(ntime,graphic)
  varray_copy = new(ntime,graphic)

  do i=0,17
    carray(i) = gsn_contour(wks,t(i,:,:),cnres)
    varray(i) = gsn_vector_scalar(wks,u(i,:,:),v(i,:,:),t(i,:,:),vcres)
  end do
  carray_copy = carray
  varray_copy = varray

  pnlres             = True
  pnlres@gsnPanelLabelBar = True
  pnlres@gsnDebug         = True
  pnlres@gsnMaximize      = True

  txres                       = True
  txres@txFontHeightF         = 0.03
  txres@txBackgroundFillColor = "white"
  txres@txPerimOn             = True
  txres@gsnFrame              = True

;
; Begin panel plots.
;
  panel_dims = new(2,integer)

  nplots = 6
  panel_dims(0) = 3
  panel_dims(1) = 2
  gsn_panel(wks, varray(0:nplots-1), panel_dims, pnlres)

  pnlres@lbOrientation  = "Vertical"
  gsn_panel(wks, varray(0:nplots-1), panel_dims, pnlres)

  gsn_panel(wks, varray(0:nplots-1), panel_dims, pnlres)

  pnlres@lbLabelStride = 2
  delete(pnlres@vpWidthF)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)

;
; Begin row_spec plots.
;
  row_spec = (/3,2,1/)

  pnlres@gsnPanelRowSpec  = True
  pnlres@gsnFrame         = False
  delete(pnlres@lbLabelStride)
  pnlres@gsnPanelCenter = True
  pnlres@lbOrientation          = "Vertical"
  pnlres@pmLabelBarOrthogonalPosF = 0.02
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)

  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(5) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(5) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:4) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:4) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  delete(pnlres@gsnPanelCenter)
  delete(pnlres@gsnPanelRowSpec)

;
; Start with a single row of plots, then make the first plot and the last
; plot missing.
;
  panel_dims(0) = 1
  panel_dims(1) = 3
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Single column of plots, then make the first plot and the last
; plot missing.
;
  panel_dims(0) = 3
  panel_dims(1) = 1
  nplots = panel_dims(0) * panel_dims(1)
;;;;
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 2 rows x 3 columns of plots. Then make the top, bottom, right,
; and left rows missing.
;
  panel_dims(0) = 2
  panel_dims(1) = 3
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  carray(3) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  carray(5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 3 rows x 2 columns of plots. Then make the top, bottom, right,
; and left rows missing.
;
  panel_dims(0) = 3
  panel_dims(1) = 2
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2:3) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(4:5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  carray(2) = carray@_FillValue
  carray(4) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(1) = carray@_FillValue
  carray(3) = carray@_FillValue
  carray(5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 3 rows x 2 columns of plots, but only pass in 4 plots, then 2, then 1.
;
  panel_dims(0) = 3
  panel_dims(1) = 2
  nplots = 4
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"4 plots passed in",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"4 plots passed in",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  nplots = 2 
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"2 plots passed in",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  nplots = 1
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"1 plot passed in",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 3 rows x 4 columns of plots, but only pass in 8, 5, 3, then 1 plot.
; On this one, we alternate between setting gsnPanelCenter to True
; and False.
;
  panel_dims(0) = 3
  panel_dims(1) = 4
  nplots = 8
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"8 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"8 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Same thing as above, only with some missing plots thrown in.
;
  carray(0:1) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"8 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"8 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Now do 5 plots.
;
  nplots = 5 
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"5 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"5 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Same thing as above, only with some missing plots thrown in.
;
  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"5 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"5 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Now 3.
;
  nplots = 3 
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"3 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"3 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Same thing as above, only with some missing plots thrown in.
;
  carray(0:1) = carray@_FillValue
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"3 plots passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"3 plots passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)


;
; One plot
;
  nplots = 1
  pnlres@gsnPanelCenter = True
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"1 plot passed in:C:center is True",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  nplots = 1
  pnlres@gsnPanelCenter = False
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"1 plot passed in:C:center is False",False)
  gsn_text_ndc(wks,title,0.5,0.5,txres)
end
