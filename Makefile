gsun: gsun.o driver.o gsun.h
	nhlcc -g -o gsun driver.o gsun.o -lnetcdf

driver.o: driver.c gsun.h
	nhlcc -g -c driver.c

gsun.o: gsun.c gsun.h
	nhlcc -g -c gsun.c

gsun_struct: gsun_struct.o driver_struct2.o gsun_struct.h
	nhlcc -g -c gsun.c

gsun_struct: gsun_struct.o driver_struct2.o gsun_struct.h
	nhlcc -g -o gsun_struct driver_struct2.o gsun_struct.o -lnetcdf

driver_struct.o: driver_struct.c gsun_struct.h
	nhlcc -g -c driver_struct.c

driver_struct2.o: driver_struct2.c gsun_struct.h
	nhlcc -g -c driver_struct2.c

gsun_struct.o: gsun_struct.c gsun_struct.h
	nhlcc -g -c gsun_struct.c

