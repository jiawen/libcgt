TEMPLATE = lib
CONFIG += qt thread c++11 debug_and_release staticlib
QT += widgets

DESTDIR = "../lib"

INCLUDEPATH += "./include"

win32 {
    CONFIG( debug, debug|release ) {
      TARGET = libcgt_cored
    } else {
      TARGET = libcgt_core
      DEFINES += _SECURE_SCL=0
    }

    QMAKE_CXXFLAGS += -MP
    DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX 
}

HEADERS += include/libcgt_core.h

HEADERS += include/cameras/*.h
SOURCES += src/cameras/*.cpp

HEADERS += include/common/*.h
HEADERS += include/common/*.inl
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
HEADERS += include/math/*.inl
SOURCES += src/math/*.cpp

HEADERS += include/time/*.h
SOURCES += src/time/*.cpp

HEADERS += include/vecmath/*.h
HEADERS += include/vecmath/*.inl
SOURCES += src/vecmath/*.cpp

# LodePNG
HEADERS += external/lodepng.h
SOURCES += external/lodepng.cpp
