CXX ?= g++
CXXFLAGS ?= -O2 -std=c++11 -pthread -I..
LDFLAGS ?= -pthread

TARGET := restOfLife
SRC := main.cc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
