TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

INCLUDEPATH += $(CUDA_INC_PATH)
INCLUDEPATH += $(NVCUDASAMPLES_ROOT)/common/inc
INCLUDEPATH += "./include"
INCLUDEPATH += "../core/include"

CONFIG( debug, debug|release ) {
  TARGET = libcgt_cudad
} else {
  TARGET = libcgt_cuda
  DEFINES += _SECURE_SCL=0
}

QMAKE_CXXFLAGS += -MP
DEFINES += _CRT_SECURE_NO_WARNINGS

HEADERS += include/*.h
HEADERS += include/*.inl
SOURCES += src/*.cpp
SOURCES += src/*.cu

