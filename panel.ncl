load "$NCARG_ROOT/lib/ncarg/nclex/gsun/gsn_code.ncl"

function create_title(parray, parray_copy, nplots, panel_dims[2])
begin
  title = new(1,string)

  first_time = True
  do i = 0,nplots-1
    if(ismissing(parray(i))) then
      if(first_time) then
        title = ":F25:" + panel_dims(0) + " rows/" + panel_dims(1) + " columns:C:plots " + i
        first_time = False
      else
        title = title + ", " + i
      end if
      parray(i) = parray_copy(i)
    end if
  end do
  if(.not.first_time) then
    title = title + " are missing"
  else
    title = ":F25:" + panel_dims(0) + " rows/" + panel_dims(1) + " columns"
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
  c = addfile(cdfpath + "Tstorm.cdf","r")

  t = c->t
  t = (t-273.15)*(9./5.) + 32.

  wks = gsn_open_wks("ps","panel2")
  gsn_define_colormap(wks,cmap)

  res = True
  res@gsnDraw                = False
  res@gsnFrame               = False
  res@sfYArray               = t&lat
  res@sfXArray               = t&lon

  res@pmLabelBarDisplayMode  = "Always"
  res@pmLabelBarSide         = "Bottom"
  res@lbOrientation          = "Horizontal"
  res@lbPerimOn              = "False"
  res@lbTitleFont            = 25
  res@lbTitleString          = "TEMPERATURE (:S:o:N:F)"
  res@tiMainFontHeightF      = 0.03
  res@tiMainString           = ":F25:Temperature"
  res@cnFillOn               = True
    
  ntime = dimsizes(t(:,0,0))

  carray      = new(ntime,graphic)
  carray_copy = new(ntime,graphic)

  do i=0,19
    carray(i) = gsn_contour(wks,t(i,:,:),res)
  end do
  carray_copy = carray

  pnlres             = True
  pnlres@gsnPanelLabelbar = True
  pnlres@gsnDebug    = True
  pnlres@gsnMaximize = True
  pnlres@gsnFrame    = False

  txres                       = True
  txres@txFontHeightF         = 0.03
  txres@txBackgroundFillColor = "white"
  txres@txPerimOn             = True
  txres@gsnFrame              = True

;
; Begin panel plots.
;
  panel_dims = new(2,integer)

;
; Start with a single row of plots, then make the first plot and the last
; plot missing.
;
  panel_dims(0) = 1
  panel_dims(1) = 3
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; Single column of plots, then make the first plot and the last
; plot missing.
;
  panel_dims(0) = 3
  panel_dims(1) = 1
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 2 rows x 3 columns of plots. Then make the top, bottom, right,
; and left rows missing.
;
  panel_dims(0) = 2
  panel_dims(1) = 3
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  carray(3) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
  carray(5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 3 rows x 2 columns of plots. Then make the top, bottom, right,
; and left rows missing.
;
  panel_dims(0) = 3
  panel_dims(1) = 2
  nplots = panel_dims(0) * panel_dims(1)
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2:3) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(4:5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  carray(2) = carray@_FillValue
  carray(4) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(1) = carray@_FillValue
  carray(3) = carray@_FillValue
  carray(5) = carray@_FillValue

  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

;
; 3 rows x 2 columns of plots, but only pass in 4 plots, then 2, then 1.
;
  panel_dims(0) = 3
  panel_dims(1) = 2
  nplots = 4
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:1) = carray@_FillValue
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  nplots = 2 
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)

  nplots = 1
  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims)
  gsn_text_ndc(wks,title,0.5,0.5,txres)
end
