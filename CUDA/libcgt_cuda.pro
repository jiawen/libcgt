TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

INCLUDEPATH += $(CUDA_INC_PATH)
INCLUDEPATH += $(NVSDKCOMPUTE_ROOT)/C/common/inc
INCLUDEPATH += "./include"
INCLUDEPATH += "../core/include"

CONFIG( debug, debug|release ) {
  TARGET = libcgt_cudad
} else {
  TARGET = libcgt_cuda
  DEFINES += _SECURE_SCL=0
}

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS
# For PhysX
DEFINES += _ITERATOR_DEBUG_LEVEL=0

HEADERS += include/*.h
HEADERS += include/*.inl
SOURCES += src/*.cpp
