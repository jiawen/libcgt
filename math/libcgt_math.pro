TEMPLATE = lib
CONFIG += qt thread debug_and_release staticlib

DESTDIR = "../lib"

# ===== libcgt_core =====
INCLUDEPATH += "../core/include"
LIBPATH += "../lib"

# ===== Qt =====
INCLUDEPATH += $(QTDIR)/include/QtCore
INCLUDEPATH += $(QTDIR)/include/QtGui
INCLUDEPATH += $(QTDIR)/include
LIBPATH += $(QTDIR)/lib

# ===== MKL =====
INCLUDEPATH += $(ICPP_COMPILER12)/mkl/include
LIBPATH += $(ICPP_COMPILER12)/mkl/lib/intel64

LIBS += mkl_core.lib mkl_intel_lp64.lib

# sequential
#LIBS += mkl_sequential.lib
# multi-threaded
LIBPATH += $(ICPP_COMPILER12)/compiler/lib/intel64
LIBS += mkl_intel_thread.lib libiomp5md.lib # libiomp5mt.lib is the static library, for /MT

INCLUDEPATH += "./include"

QMAKE_CXXFLAGS += -MP4
DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG( debug, debug|release ) {
  TARGET = libcgt_mathd
  LIBS += libcgt_cored.lib

  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparsed/UFconfig"
  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparsed/CHOLMOD/Include"
  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparsed/SPQR/Include"

  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/AMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/CAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/COLAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/CCOLAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/CHOLMOD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparsed/SPQR/Lib"

  LIBS += libamd.lib libcamd.lib libcolamd.lib libccolamd.lib libcholmod.lib libspqr.lib

} else {
  TARGET = libcgt_math
  DEFINES += _SECURE_SCL=0
  LIBS += libcgt_core.lib

  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparse/UFconfig"
  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparse/CHOLMOD/Include"
  INCLUDEPATH += "C:/work/libs/cpp/SuiteSparse/SPQR/Include"

  LIBPATH += "C:/work/libs/cpp/SuiteSparse/AMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparse/CAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparse/COLAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparse/CCOLAMD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparse/CHOLMOD/Lib"
  LIBPATH += "C:/work/libs/cpp/SuiteSparse/SPQR/Lib"

  LIBS += libamd.lib libcamd.lib libcolamd.lib libccolamd.lib libcholmod.lib libspqr.lib
}

HEADERS += include/*.h
SOURCES += src/*.cpp
