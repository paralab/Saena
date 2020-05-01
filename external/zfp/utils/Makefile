include ../Config

TARGET = ../bin/zfp

all: $(TARGET)

$(TARGET): zfp.c ../lib/$(LIBZFP)
	mkdir -p ../bin
ifeq ($(WITH_IPP), yes)
	$(CC) $(CFLAGS) zfp.c -L../lib -lzfp -lm -o $(TARGET) $(IPPLIBS)
else
	$(CC) $(CFLAGS) zfp.c -L../lib -lzfp -lm -o $(TARGET)
endif

clean:
	rm -f $(TARGET) fields.o
