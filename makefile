CC = gcc

CFLAGS = -std=c99

SOURCES = main.c \
					value.c \
					memory.c \
					debug.c \
					chunk.c \
					

HEADERS = common.h \
					value.h \
					memory.h \
					chunk.h \
					debug.h \


OUTPUT = clox

$(OUTPUT) : $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(OUTPUT) $(SOURCES)

clean: 
	rm -f $(OUTPUT)
