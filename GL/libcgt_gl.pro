TEMPLATE = lib
CONFIG += qt thread c++11 debug_and_release staticlib
QT += widgets opengl

DESTDIR = "../lib"

INCLUDEPATH += "./include"
INCLUDEPATH += "../core/include"
INCLUDEPATH += $$(GLEW_INCLUDE_PATH)

win32 {
    CONFIG( debug, debug|release ) {
      TARGET = libcgt_gld
    } else {
      TARGET = libcgt_gl
      DEFINES += _SECURE_SCL=0
    }

    QMAKE_CXXFLAGS += -MP
    DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX 
}

HEADERS += include/*.h
SOURCES += src/*.cpp
