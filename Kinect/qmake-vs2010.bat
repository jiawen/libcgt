%QTDIR%\bin\qmake -tp vc -spec win32-msvc2010 -o libcgt_kinect.vcxproj libcgt_kinect.pro
fsi --exec ..\vcxproj_win32tox64.fsx libcgt_kinect.vcxproj
