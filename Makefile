gsun: gsun.o driver.o gsun.h
	nhlcc -o gsun driver.o gsun.o -lnetcdf

driver.o: driver.c
	nhlcc -c driver.c

gsun.o: gsun.c
	nhlcc -c gsun.c

