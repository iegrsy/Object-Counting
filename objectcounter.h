#ifndef OBJECTCOUNTER_H
#define OBJECTCOUNTER_H

#include <QImage>
#include <QDebug>
#include <QTime>

#include "opencv2/video/background_segm.hpp"
#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <highgui.h>
#include <cv.h>

#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;

class ObjectCounter
{
public:
	ObjectCounter();

	void imgShow(QImage img);
	void movemontDetection(const Mat &img);
	void init();
private:
	Mat frame1, knn;
	Mat grayImage1,grayImage2;
	Mat differenceImage, thresholdImage;
	Ptr<BackgroundSubtractor> pKNN;
	bool isDebugmod, isCountmod, isDrawingmod, isSettingmod;
};

#endif // OBJECTCOUNTER_H
