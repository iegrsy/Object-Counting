QT += core gui network printsupport

TARGET = ObjectCounting
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    objectcounter.cpp

HEADERS += \
    objectcounter.h

INCLUDEPATH += /usr/local/include/opencv

LIBS += -L/usr/local/lib \
-lopencv_ml \
-lopencv_objdetect \
-lopencv_shape\
-lopencv_stitching\
-lopencv_superres\
-lopencv_videostab\
-lopencv_calib3d\
-lopencv_features2d\
-lopencv_highgui\
-lopencv_videoio\
-lopencv_imgcodecs\
-lopencv_video\
-lopencv_photo\
-lopencv_imgproc\
-lopencv_flann\
-lopencv_core
