;load "$NCARG_ROOT/lib/ncarg/nclex/gsun/gsn_code.ncl"
load "./gsn_code.ncl"

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
  c = addfile(cdfpath + "Tstorm.cdf","r")

  t = c->t
  t = (t-273.15)*(9./5.) + 32.

  wks = gsn_open_wks("x11","panel2")

  res = True
  res@gsnDraw                = False
  res@gsnFrame               = False
  res@gsnSpreadColors        = True

  if(isatt(res,"gsnSpreadColors").and.res@gsnSpreadColors) then
    gsn_define_colormap(wks,"rainbow")
  else
    gsn_define_colormap(wks,cmap)
  end if

  res@sfYArray               = t&lat
  res@sfXArray               = t&lon

  res@pmLabelBarDisplayMode  = "Always"
  res@pmLabelBarSide         = "Bottom"
  res@lbOrientation          = "Horizontal"
  res@lbPerimOn              = "False"
  res@lbTitleFont            = 25
  res@lbTitleString          = "temperature (:S:o:N:F)"
  res@pmLabelBarDisplayMode  = "Never"
  res@tiMainFontHeightF      = 0.03
  res@tiMainString           = ":F25:Wind velocity vectors"
  res@cnFillOn               = True
  res@cnLevelSelectionMode   = "ManualLevels"
  res@cnMinLevelValF         = -20.
  res@cnMaxLevelValF         =  80.
  res@cnLevelSpacingF        =  10.

  ntime = dimsizes(t(:,0,0))

  carray      = new(ntime,graphic)
  carray_copy = new(ntime,graphic)

  do i=0,17
    carray(i) = gsn_contour(wks,t(i,:,:),res)
  end do
  carray_copy = carray

  pnlres             = True
  pnlres@gsnPanelLabelBar = True
  pnlres@gsnPanelRowSpec  = True
  pnlres@gsnDebug    = True
  pnlres@gsnMaximize = True
  pnlres@gsnFrame    = False

  txres                       = True
  txres@txFontHeightF         = 0.03
  txres@txBackgroundFillColor = "white"
  txres@txPerimOn             = True
  txres@gsnFrame              = True

;
; Begin row_spec plots.
;
  row_spec = (/3,2,1/)

  nplots = 6
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue
  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(5) = carray@_FillValue
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(5) = carray@_FillValue
  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:2) = carray@_FillValue
  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(3:4) = carray@_FillValue
  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:4) = carray@_FillValue
  pnlres@gsnPanelCenter = True
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is True",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0:4) = carray@_FillValue
  pnlres@gsnPanelCenter = False
;  gsn_panel(wks, carray(0:nplots-1), row_spec, pnlres)
  title = create_title(carray, carray_copy, nplots, row_spec,"PanelCenter is False",True)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  delete(pnlres@gsnPanelCenter)
  delete(pnlres@gsnPanelRowSpec)
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
;  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(0) = carray@_FillValue

;  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

  carray(2) = carray@_FillValue
;  gsn_panel(wks, carray(0:nplots-1), panel_dims, pnlres)
  title = create_title(carray, carray_copy, nplots, panel_dims,"",False)
;  gsn_text_ndc(wks,title,0.5,0.5,txres)

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
