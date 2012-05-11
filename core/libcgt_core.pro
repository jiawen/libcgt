TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

CONFIG( debug, debug|release ) {
  TARGET = libcgt_cored
} else {
  TARGET = libcgt_core
  DEFINES += _SECURE_SCL=0
}

INCLUDEPATH += $(QTDIR)/include/QtCore
INCLUDEPATH += $(QTDIR)/include/QtGui
INCLUDEPATH += $(QTDIR)/include

INCLUDEPATH += "./include"

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX 
# For PhysX
DEFINES += _ITERATOR_DEBUG_LEVEL=0

HEADERS += include/libcgt_core.h

HEADERS += include/cameras/*.h
SOURCES += src/cameras/*.cpp

HEADERS += include/color/*.h
SOURCES += src/color/*.cpp

HEADERS += include/common/*.h
SOURCES += src/common/*.cpp

HEADERS += include/geometry/*.h
SOURCES += src/geometry/*.cpp

HEADERS += include/imageproc/*.h
SOURCES += src/imageproc/*.cpp

HEADERS += include/io/*.h
SOURCES += src/io/*.cpp

HEADERS += include/lights/*.h
SOURCES += src/lights/*.cpp

HEADERS += include/math/*.h
SOURCES += src/math/*.cpp

HEADERS += include/time/*.h
SOURCES += src/time/*.cpp

HEADERS += include/vecmath/*.h
SOURCES += src/vecmath/*.cpp
