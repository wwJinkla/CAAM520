### Begin MACROS
#Define compiler
CC=gcc -fopenmp
#Define linker
LD=gcc -fopenmp
#Compilation flags
CFLAGS= -c 
#Link flags
LDFLAGS=
#Define addition librariry (like -lm)
LIBS=-lm
#Define include path (where header file could be)
INCLUDE=./
#Define sources files to compile
SOURCES=hw03.c 
#SOURCES=$(wildcard *.c)
#Define generated object files from SOURCES
OBJECTS=$(SOURCES:.c=.o)
#Define name executable
EXEC=hw03
### End MACROS

### Begin targets
build: $(EXEC) $(SOURCES)

$(EXEC) : $(OBJECTS)
	$(LD) $(LDFLAGS) $(OBJECTS) $(LIBS) -o $@
%.o: %.c
	$(CC) $(CFLAGS) -I$(INCLUDE) $< -o $@

#Define additional action
clean :
	rm -rf $(OBJECTS)

all : clean build
### End targets
