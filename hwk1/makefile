### Begin MACROS
#Define compiler
CC=gcc
#Define linker
LD=gcc
#Compilation flags
CFLAGS= -c -std=c99 -Wall
#Link flags
LDFLAGS=
#Define addition librariry (like -lm)
LIBS=-lm
#Define include path (where header file could be)
INCLUDE=./
#Define sources files to compile
SOURCES=main.c 
#SOURCES=$(wildcard *.c)
#Define generated object files from SOURCES
OBJECTS=$(SOURCES:.c=.o)
#Define name executable
EXEC=hw01
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
