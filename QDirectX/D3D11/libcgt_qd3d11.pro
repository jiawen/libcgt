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

CONFIG( debug, debug|release ) {
  TARGET = libcgt_qd3d11d
} else {
  TARGET = libcgt_qd3d11
  DEFINES += _SECURE_SCL=0
}

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX _ITERATOR_DEBUG_LEVEL=0

# Code
HEADERS = include/*.h
SOURCES = src/*.cpp
