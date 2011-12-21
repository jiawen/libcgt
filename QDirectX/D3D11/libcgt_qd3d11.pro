TEMPLATE = lib
CONFIG += qt console thread debug_and_release staticlib

DESTDIR = "../../lib"

INCLUDEPATH += ./include

INCLUDEPATH += $(QTDIR)/include/QtCore
INCLUDEPATH += $(QTDIR)/include/QtGui
INCLUDEPATH += $(QTDIR)/include

INCLUDEPATH += $(DXSDK_DIR)Include
INCLUDEPATH += $(DXSDK_DIR)Samples/C++/Effects11/Inc

INCLUDEPATH += ../../core/include

LIBPATH += $(QTDIR)/lib
LIBPATH += $(DXSDK_DIR)Lib/x64

#LIBS += d3d11.lib d3dcompiler.lib Effects11.lib

CONFIG( debug, debug|release ) {
  LIBPATH += $(DXSDK_DIR)Samples/C++/Effects11/x64/Debug
  #LIBPATH += ../libcgt/debug
  #LIBS += libcgt_cored.lib d3dx10d.lib d3dx11d.lib 
  #LIBS += libcgt_cored.lib d3dx11d.lib 
  TARGET = libcgt_qd3d11d
} else {
  LIBPATH += $(DXSDK_DIR)Samples/C++/Effects11/x64/Release
  #LIBPATH += ../libcgt/release
  #LIBS += libcgt_core.lib d3dx10.lib d3dx11.lib 
  #LIBS += libcgt_core.lib d3dx11.lib 
  TARGET = libcgt_qd3d11
  DEFINES += _SECURE_SCL=0
}

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS

# Code
HEADERS = include/*.h
SOURCES = src/*.cpp
