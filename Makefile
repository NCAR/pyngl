gsun: gsun.c gsun.h
	nhlcc -o gsun gsun.c -lnetcdf

pygsun: pygsun.o driver.o gsun.h
	nhlcc -o pygsun driver.o pygsun.o -lnetcdf

driver.o: driver.c
	nhlcc -c driver.c

pygsun.o: pygsun.c
	nhlcc -c pygsun.c

