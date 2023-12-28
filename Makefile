CC = nvcc
CFLAGS = --gpu-architecture=compute_75

TARGET = galaxy
SRCDIR = .
INPUTDIR = input
REAL_INPUT = $(INPUTDIR)/input-100000-real.txt
SYNTHETIC_INPUT = $(INPUTDIR)/input-100000-syntethic.txt

all: clean build run

build:
	$(CC) $(CFLAGS) $(SRCDIR)/$(TARGET).cu -o $(TARGET)

run:
	time ./$(TARGET) $(REAL_INPUT) $(SYNTHETIC_INPUT)

clean:
	rm -f $(TARGET)
