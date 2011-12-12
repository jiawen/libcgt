TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

# libcgt_core
INCLUDEPATH += "../core/include"

# Qt
INCLUDEPATH += $(QTDIR)/include/QtCore
INCLUDEPATH += $(QTDIR)/include/QtGui
INCLUDEPATH += $(QTDIR)/include
LIBPATH += $(QTDIR)/lib

# MKL
INCLUDEPATH += $(ICPP_COMPILER12)/mkl/include
LIBPATH += $(ICPP_COMPILER12)/mkl/lib/intel64

LIBS += mkl_core.lib mkl_intel_lp64.lib

# sequential
#LIBS += mkl_sequential.lib
# multi-threaded
LIBPATH += $(ICPP_COMPILER12)/compiler/lib/intel64
LIBS += mkl_intel_thread.lib libiomp5md.lib

INCLUDEPATH += "./include"

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG( debug, debug|release ) {
  TARGET = libcgt_mathd
  LIBS += "../core/debug/libcgt_cored.lib"
} else {
  TARGET = libcgt_math
  DEFINES += _SECURE_SCL=0
  LIBS += "../core/release/libcgt_core.lib"
}


HEADERS += include/*.h
SOURCES += src/*.cpp
