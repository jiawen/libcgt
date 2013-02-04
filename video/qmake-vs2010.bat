%QTDIR%\bin\qmake -tp vc -spec win32-msvc2010 -o libcgt_video.vcxproj libcgt_video.pro
fsi --exec ..\vcxproj_win32tox64.fsx libcgt_video.vcxproj
