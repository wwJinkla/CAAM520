all:	cfd

cfd: hw04.o
	nvcc main.o -o cfd

hw04.o:	hw04.cu
	nvcc -c hw04.cu -o hw04.o -lm


clean:
	rm -f *.o cfd	
