gsun: gsun.o driver.o gsun.h
	nhlcc -g -o gsun driver.o gsun.o -lnetcdf

driver.o: driver.c gsun.h
	nhlcc -g -c driver.c

gsun.o: gsun.c gsun.h
	nhlcc -g -c gsun.c

