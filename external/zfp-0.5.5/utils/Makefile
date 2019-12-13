include ../Config

TARGET = ../bin/send_zfp

all: $(TARGET)

$(TARGET): send_zfp.c ../lib/$(LIBZFP)
	mkdir -p ../bin
	$(CC) $(CFLAGS) send_zfp.c -L../lib -lzfp -lm -o $(TARGET)

clean:
	rm -f $(TARGET) fields.o
