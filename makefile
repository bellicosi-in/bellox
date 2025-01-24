CC = gcc

CFLAGS = -std=c99

SOURCES = main.c \
					value.c \
					memory.c \
					debug.c \
					chunk.c \
					vm.c \
					compiler.c \
					scanner.c \
					object.c \
					table.c \
					

HEADERS = common.h \
					value.h \
					memory.h \
					chunk.h \
					debug.h \
					vm.h \
					compiler.h \
					scanner.h \
					object.h \
					table.h \


OUTPUT = clox

$(OUTPUT) : $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(OUTPUT) $(SOURCES)

clean: 
	rm -f $(OUTPUT)
