# use with x64 native tools commands prompt for vs (nmake).

TARGET = rest_of_life_cuda.exe
CPP_SOURCES = main.cc device_scene.cc scene_builder.cc
CU_SOURCES = renderer.cu
OBJECTS = $(CPP_SOURCES:.cc=.obj)
CU_OBJECTS = $(CU_SOURCES:.cu=.obj)

!IFNDEF MSVC_BIN
MSVC_BIN = C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64
!ENDIF

CXX = "$(MSVC_BIN)\cl.exe"
CXXFLAGS = /std:c++17 /O2 /EHsc /MD /I.. /D_USE_MATH_DEFINES /DMAXFLOAT=3.402823466e+38F
!IF DEFINED(CUDA_PATH)
CXXFLAGS = $(CXXFLAGS) /I"$(CUDA_PATH)\include"
!ENDIF
NVCC = nvcc
NVCCFLAGS = -std=c++17 -O2 -I.. -DMAXFLOAT=3.402823466e+38F -D_USE_MATH_DEFINES -Xcompiler "/EHsc /MD" -ccbin "$(MSVC_BIN)"
LDFLAGS =
!IF DEFINED(CUDA_PATH)
LDFLAGS = /link /LIBPATH:"$(CUDA_PATH)\lib\x64" cudart.lib cudadevrt.lib
!ENDIF

.SUFFIXES: .cc .obj

.cc.obj:
	$(CXX) $(CXXFLAGS) /c $< /Fo$@

build: $(TARGET)

$(TARGET): $(OBJECTS) $(CU_OBJECTS)
	$(CXX) $(OBJECTS) $(CU_OBJECTS) /Fe$@ $(LDFLAGS)

renderer.obj: renderer.cu
	$(NVCC) $(NVCCFLAGS) -c renderer.cu -o renderer.obj

clean:
	@if exist $(TARGET) del $(TARGET)
	@for %%f in ($(OBJECTS)) do @if exist %%f del %%f
	@for %%f in ($(CU_OBJECTS)) do @if exist %%f del %%f
