TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

# ===== libcgt_video =====
INCLUDEPATH += "../core/include"
INCLUDEPATH += "./include"

# FFMPEG
# HACK
INCLUDEPATH += ../../libs/include/ffmpeg

QMAKE_CXXFLAGS += -MP
DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX

CONFIG( debug, debug|release ) {
  TARGET = libcgt_videod

} else {
  TARGET = libcgt_video
  DEFINES += _SECURE_SCL=0
}

HEADERS += include/*.h
SOURCES += src/*.cpp
