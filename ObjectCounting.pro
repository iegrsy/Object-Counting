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
-lopencv_stitching \
-lopencv_superres \
-lopencv_videostab \
-lopencv_photo \
-lopencv_aruco \
-lopencv_bgsegm \
-lopencv_bioinspired \
-lopencv_ccalib \
-lopencv_cvv \
-lopencv_dpm \
-lopencv_face \
-lopencv_freetype \
-lopencv_fuzzy \
-lopencv_hdf \
-lopencv_img_hash \
-lopencv_line_descriptor \
-lopencv_optflow \
-lopencv_reg \
-lopencv_rgbd \
-lopencv_saliency \
-lopencv_stereo \
-lopencv_structured_light \
-lopencv_phase_unwrapping \
-lopencv_surface_matching \
-lopencv_tracking \
-lopencv_datasets \
-lopencv_text \
-lopencv_dnn \
-lopencv_plot \
-lopencv_xfeatures2d \
-lopencv_shape \
-lopencv_video \
-lopencv_ml \
-lopencv_ximgproc \
-lopencv_calib3d \
-lopencv_features2d \
-lopencv_highgui \
-lopencv_videoio \
-lopencv_flann \
-lopencv_xobjdetect \
-lopencv_imgcodecs \
-lopencv_objdetect \
-lopencv_xphoto \
-lopencv_imgproc \
-lopencv_core
