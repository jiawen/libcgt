TEMPLATE = lib
CONFIG += qt console thread debug_and_release staticlib

DESTDIR = "../lib"

INCLUDEPATH += ./include

INCLUDEPATH += $(QTDIR)/include/QtCore
INCLUDEPATH += $(QTDIR)/include/QtGui
INCLUDEPATH += $(QTDIR)/include

INCLUDEPATH += $(KINECTSDK10_DIR)inc

INCLUDEPATH += ../core/include

CONFIG( debug, debug|release ) {
  TARGET = libcgt_kinectd
} else {
  TARGET = libcgt_kinect
  DEFINES += _SECURE_SCL=0
}

QMAKE_CXXFLAGS += -MP
DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX

# Code
HEADERS = include/*.h
SOURCES = src/*.cpp
