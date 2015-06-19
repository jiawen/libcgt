TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

# ===== libcgt_core =====
INCLUDEPATH += "../core/include"

# ===== MKL =====
INCLUDEPATH += $(ICPP_COMPILER13)mkl/include

INCLUDEPATH += "./include"

QMAKE_CXXFLAGS += -MP
DEFINES += _CRT_SECURE_NO_WARNINGS NOMINMAX

CONFIG( debug, debug|release ) {
  TARGET = libcgt_mathd

  INCLUDEPATH += $(SUITESPARSED)/UFconfig
  INCLUDEPATH += $(SUITESPARSED)/CHOLMOD/Include
  INCLUDEPATH += $(SUITESPARSED)/SPQR/Include

} else {
  TARGET = libcgt_math
  DEFINES += _SECURE_SCL=0

  INCLUDEPATH += $(SUITESPARSE)/UFconfig
  INCLUDEPATH += $(SUITESPARSE)/CHOLMOD/Include
  INCLUDEPATH += $(SUITESPARSE)/SPQR/Include
}

HEADERS += include/*.h
SOURCES += src/*.cpp
