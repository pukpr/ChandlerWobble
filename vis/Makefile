CXX = g++
CC = gcc
GZ_VER = gz-sim8 # Update to gz-sim7 if needed

# Discovery
GZ_CFLAGS = $(shell pkg-config --cflags $(GZ_VER))
GZ_LIBS = $(shell pkg-config --libs $(GZ_VER))
LIBS = -lrt -lpthread -lm

all: libTablePlugin.so test_producer

libTablePlugin.so: TableControlPlugin.cc shared_structs.h
	$(CXX) -shared -fPIC TableControlPlugin.cc -o libTablePlugin.so $(GZ_CFLAGS) $(GZ_LIBS) $(LIBS)

test_producer: test_producer.c shared_structs.h
	$(CC) test_producer.c -o test_producer $(LIBS)

clean:
	rm -f libTablePlugin.so test_producer
