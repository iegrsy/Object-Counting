#include "objectcounter.h"

static int SENSITIVITY_VALUE = 150;
static int BLUR_SIZE = 15;
static int CLOSE_VALUE = 80;
static int MIN_AREA = 1000;

static bool isFirst = false;
static int s_slider = SENSITIVITY_VALUE;
static int b_slider = BLUR_SIZE;
static int c_slider = CLOSE_VALUE;
static int m_slider = MIN_AREA;

static int slider_max = 200;
static void on_trackbar(int, void*){
	SENSITIVITY_VALUE = s_slider;
	BLUR_SIZE = b_slider;
	CLOSE_VALUE = c_slider;
	MIN_AREA = m_slider;
}
static void mypause(){
	Q_UNUSED(mypause)
	while (true)
		if( (char)waitKey(10) == 'p' )
			break;
}

static Rect countRect;
static Point rectStart;
static Point rectEnd;
static Point lineStart;
static Point lineEnd;

static int objectUpCount;
static int objectDownCount;

static int isPosition(Point a, Point b, Point c){
	int s = ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x));
	if(s > 0)
		return -1;
	else if(s == 0)
		return 0;
	else if(s < 0)
		return 1;
	else
		return 2;
}

class _objectFollow{
public:
	_objectFollow(){
		lastkey = 0;
		loopCount = 0;
	}

	void setPoint(Point mp, Mat mt){
		if(objects.isEmpty()){
			addObject();
			addObjectPoint(lastkey, mp, mt);
		}
		else{
			int ck = findCloseObject(mp, mt);
			if (objects.contains(ck))
				addObjectPoint(ck, mp, mt);
			//qDebug()<<QString("size: %1 === key: %2").arg(objects.size()).arg(ck);
		}
	}

	void clearObjects(){
		//qDebug()<<"Clearing objects.";
		objects.clear();
		objectsState.clear();
		objectsPosState.clear();
		objectsLastMat.clear();
		lastkey = 0;
	}

	void drawFootprints(Mat &img){
		QHash<int, QList<Point> >::iterator i;
		for(i = objects.begin(); i != objects.end(); ++i){
			vector< vector<Point> > printVector;
			printVector.push_back(i.value().toVector().toStdVector());
			polylines(img, printVector, false, Scalar((i.key()*28%75), ((i.key()*28%255)), ((i.key()*28%10+200))), 2, CV_AA);
		}
	}


private:
	QHash<int, QList<Point> > objects;
	QHash<int, bool> objectsState;
	QHash<int, int> objectsPosState;
	QHash<int, Mat> objectsLastMat;

	int lastkey;
	int loopCount;

	void addObject(){
		QList<Point> tp;
		objects.insert(++lastkey, tp);
		objectsState.insert(lastkey, false);
		objectsPosState.insert(lastkey, 2);
	}

	void addObjectPoint(int key, Point mp, Mat mt){
		QList<Point> tp = objects.value(key);
		tp.append(mp);
		if(tp.size() > 100)
			tp.removeFirst();

		objects.insert(key,tp);
		objectsState.insert(key, true);
		Mat m;
		mt.copyTo(m);
		objectsLastMat.insert(key, m);

		lineCount();
		clearHistory();
	}

	int getObjectCount(){
		return objects.size();
	}

	int findCloseObject(Point mp, Mat mt){
		//matching images
		if(getObjectCount() > 0){
			int oi = -1;
			int tmp = 0;
			QHashIterator<int, QList<Point> > i(objects);
			while(i.hasNext()){
				i.next();

				int dispt = compareImage(objectsLastMat.value(i.key()), mt);
				tmp = dispt;

				if(tmp == 1){
					oi = i.key();
					break;
				}
			}

			if(tmp == -1){
				addObject();
				addObjectPoint(lastkey, mp, mt);
				oi = lastkey;
			}

			return oi;
		}
		return -1;
	}

	void clearHistory(){
		loopCount++;
		if(loopCount > 50){
			QHash<int, bool>::iterator i;
			for (i = objectsState.begin(); i != objectsState.end(); ++i)
				if(!i.value())
					objects.remove(i.key());

			QHash<int, bool>::iterator i1;
			for (i1 = objectsState.begin(); i1 != objectsState.end(); ++i1)
				objectsState.insert(i1.key(),false);

			loopCount = 0;
		}
	}

	void lineCount(){
		QHashIterator<int, QList<Point> > i(objects);
		while(i.hasNext()){
			i.next();
			if(i.value().size() > 1)
				if(countRect.contains(i.value().first())){
					int p = isPosition(lineStart, lineEnd, i.value().last());
					int os = objectsPosState.value(i.key());
					//qDebug()<<QString("p: %1 s: %2").arg(p).arg(os);
					if(os == 2)
						objectsPosState.insert(i.key(), p);
					else if (os == 1 && p == -1){
						//up count
						objectDownCount++;
						objectsPosState.insert(i.key(), p);
					}else if (os == -1 && p == 1){
						//down count
						objectUpCount++;
						objectsPosState.insert(i.key(), p);
					}
				}
		}
	}

	int compareImage(Mat img_1, Mat img_2){

		Ptr<SiftFeatureDetector> siftDetector = SiftFeatureDetector::create();

		vector<KeyPoint> siftKeypoints, siftKeypoints1;
		Mat descriptors_1, descriptors_2;

		siftDetector->detectAndCompute(img_1, Mat(), siftKeypoints, descriptors_1);
		siftDetector->detectAndCompute(img_2, Mat(), siftKeypoints1, descriptors_2);

		if(descriptors_1.size().height < 1 || descriptors_1.size().width < 1 ||
				descriptors_2.size().height < 1 || descriptors_2.size().width < 1 )
			return -1;

		FlannBasedMatcher matcher;
		vector< DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );
		double max_dist = 0; double min_dist = CLOSE_VALUE;
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}

		vector< DMatch > good_matches;
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ if( matches[i].distance <= max(2*min_dist, 0.02) )
			{ good_matches.push_back( matches[i]); }
		}

		if(good_matches.size()>3)
			return 1;
		else
			return -1;
	}
};

static _objectFollow _ofollow;

void onmouse(int event, int x, int y, int flags, void* param){
	Q_UNUSED(param)
	Q_UNUSED(flags)

	if(event == CV_EVENT_LBUTTONDOWN){
		rectStart = Point(x, y);
		rectEnd = rectStart;
	}else if (event == CV_EVENT_LBUTTONUP){
		rectEnd = Point(x, y);
		if(false){
			int x1 = (rectStart.x + rectEnd.x)*.5;
			lineStart = Point(x1, rectStart.y);
			lineEnd = Point(x1, rectEnd.y);
		}else{
			int y1 = (rectStart.y + rectEnd.y)*.5;
			lineStart = Point(rectStart.x, y1);
			lineEnd = Point(rectEnd.x, y1);
		}
	}
}
static QImage Mat2QImage(cv::Mat const& src){
	Q_UNUSED(Mat2QImage)
	cv::Mat temp; // make the same cv::Mat
	cvtColor(src, temp,CV_BGR2RGB); // cvtColor Makes a copt, that what i need
	QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
	dest.bits(); // enforce deep copy, see documentation
	// of QImage::QImage ( const uchar * data, int width, int height, Format format )
	return dest;
}

static cv::Mat QImage2Mat(QImage const& src){
	cv::Mat tmp(src.height(),src.width(),CV_8UC3,(uchar*)src.bits(),src.bytesPerLine());
	cv::Mat result; // deep copy just in case (my lack of knowledge with open cv)
	cvtColor(tmp, result,CV_BGR2RGB);
	return result;
}

static void drawTarget(Point target, Mat &cameraFeed, int i){
	int x = target.x;
	int y = target.y;
	line(cameraFeed, Point(x, y + 25), Point(x, y - 25), Scalar(0, 0, 0), 2);
	line(cameraFeed, Point(x + 25, y), Point(x - 25, y), Scalar(0, 0, 0), 2);
	putText(cameraFeed,
			QString::number(i).toStdString() + "(" + QString::number(x).toStdString() + "," + QString::number(y).toStdString() + ")",
			Point(x - 38, y - 35), 1, 1, Scalar(76, 255, 0), 2);
}

ObjectCounter::ObjectCounter(){
	init();
}

void ObjectCounter::init(){
	pKNN = createBackgroundSubtractorKNN(
				300,	// history
				100.0,	// treshold
				true);

	isDebugmod = false;
	isSettingmod = true;
	isCountmod = true;
	isDrawingmod = true;
}

void ObjectCounter::imgShow(QImage img){
	frame1 = QImage2Mat(img);
	resize(frame1,frame1,Size(),0.5,0.5);
	imshow("Frame", frame1);
}

void ObjectCounter::movemontDetection(const Mat &img){
	frameOriginal = img;
	frameOriginal.copyTo(frame1);
	//resize(frame1,frame1,Size(),0.5,0.5);
	pKNN->apply(frame1,knn);

	if(isDebugmod)
		imshow("Movemont Detection: KNN", knn);

	blur(knn, knn, Size(BLUR_SIZE, BLUR_SIZE));
	threshold(knn, knn, SENSITIVITY_VALUE, 255, THRESH_BINARY);
	knn.convertTo(knn, CV_8U);
	if(isDebugmod)
		imshow("Movemont Detection: KNN2", knn);

	vector< vector<Point> > contours;
	findContours( knn, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE );

	int cSize = contours.size();
	if (!contours.empty()){
		vector<Moments> contour_moments(cSize);
		vector<Point> mass_centers(cSize);

		bool in = false;
		for (int i = 0; i < cSize; ++i)	{
			double area = contourArea(contours[i]);
			if (area > MIN_AREA){
				contour_moments[i] = moments(contours[i], false);
				mass_centers[i] = Point(contour_moments[i].m10 / contour_moments[i].m00, contour_moments[i].m01 / contour_moments[i].m00);

				// Draw target
				Rect roi = boundingRect(contours[i]);
				//drawContours(frame1, contours, i, Scalar(0, 0, 255));
				rectangle(frame1, roi, Scalar(0, 0, 255));
				drawTarget(mass_centers[i],frame1,i);

				if(countRect.contains(mass_centers[i])){
					if(isCountmod){
						// Draw footprint
						Mat car = frameOriginal(roi);
						_ofollow.setPoint(mass_centers[i], car);
						if(isDrawingmod)
							_ofollow.drawFootprints(frame1);
					}
					in = true;
				}
			}
		}
		if(!in)
			_ofollow.clearObjects();
	}else{
		_ofollow.clearObjects();
	}

	putText(frame1, QString::number(cSize).toStdString(), cvPoint(frame1.cols-30, 30),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
	putText(frame1, QString("Up count: %1").arg(QString::number(objectUpCount)).toStdString(), Point(20,30),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
	putText(frame1, QString("Down count: %1").arg(QString::number(objectDownCount)).toStdString(), Point(20,60),
			FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
	if(isCountmod){
		countRect = Rect(rectStart,rectEnd);
		rectangle(frame1, rectStart, rectEnd, Scalar(76,255,0), 3);
		line(frame1, lineStart, lineEnd, Scalar(76,10,255), 3);
	}

	imshow("Movemont Detection", frame1);

	if(!isFirst){
		if(isSettingmod){
			createTrackbar("SENSITIVITY_VALUE", "Movemont Detection", &s_slider, slider_max, on_trackbar);
			createTrackbar("BLUR_SIZE", "Movemont Detection", &b_slider, slider_max, on_trackbar);
			createTrackbar("CLOSE_VALUE", "Movemont Detection", &c_slider, 150, on_trackbar);
			createTrackbar("MIN_AREA", "Movemont Detection", &m_slider, (int) (frame1.rows * frame1.cols), on_trackbar);
		}
		if(isCountmod)
			setMouseCallback("Movemont Detection", onmouse, &frame1);

		isFirst = true;
	}
}
