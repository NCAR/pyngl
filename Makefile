gsun: gsun.o driver.o gsun.h
	nhlcc -g -o gsun driver.o gsun.o -lnetcdf

driver.o: driver.c
	nhlcc -g -c driver.c

gsun.o: gsun.c
	nhlcc -g -c gsun.c

