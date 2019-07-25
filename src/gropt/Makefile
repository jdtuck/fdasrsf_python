.PHONY : clean

OS := $(shell uname)
ERR = $(shell which icpc>/dev/null; echo $$?)
ifeq "$(OS)" "Darwin"
	ifeq "$(ERR)" "0"
		CXX = icpc
		CFLAGS = -fPIC -std=c++11 -O3 -g -mkl
		LDFLAGS = -shared
		LIBS = -lmkl_rt -lpthread
	else
		CFLAGS =  -fPIC -std=c++11 -O3 -g  # C flags
        	LDFLAGS = -shared # linking flags
        	LIBS = -lblas -llapack
	endif
else
	ifeq "$(ERR)" "0"
    		CXX = icpc
		CFLAGS =  -fPIC -O3 -g # C flags
		LDFLAGS = -shared # linking flags
		LIBS = -lmkl_rt -lpthread
	else
		CFLAGS =  -fPIC -std=c++11 -O3 -g  -L./destdir/lib # C flags
		LDFLAGS = -shared # linking flags
		LIBS = -lopenblas64_
	endif
endif

LIB=gropt
SUFFIX=so
SOURCES = $(shell echo src/*.cpp)
HEADERS = $(shell echo incl/*.h)
OBJECTS = $(SOURCES:.cpp=.o)
INC = -Iincl/
TARGET=$(LIB).$(SUFFIX)

all: $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

install:
	cp $(TARGET) ../

src/%.o: src/%.cpp
	$(CXX) $(CFLAGS) $(INC) -c -o $@ $<

$(TARGET) : $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
