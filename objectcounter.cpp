#include "objectcounter.h"

static int SENSITIVITY_VALUE = 150;
static int BLUR_SIZE = 15;
static int CLOSE_VALUE = 2500;

static bool isFirst = false;
static int s_slider = SENSITIVITY_VALUE;
static int b_slider = BLUR_SIZE;
static int c_slider = CLOSE_VALUE;
static int slider_max = 200;
static void on_trackbar(int, void*){
	SENSITIVITY_VALUE = s_slider;
	BLUR_SIZE = b_slider;
	CLOSE_VALUE = c_slider;
}
static void mypause(){
	Q_UNUSED(mypause)
	while (true)
		if( (char)waitKey(10) == 'p' )
			break;
}

static int minArea = 200;

static Rect countRect;
static Point rectStart;
static Point rectEnd;
static Point lineStart;
static Point lineEnd;

static int objectUpCount;
static int objectDownCount;
static QList<Scalar> clr;

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
static bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r){
	Q_UNUSED(intersection)

	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x)/cross;
	r = o1 + d1 * t1;
	return true;
}

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
class _objectTracking{
public:
	//tracking methods: {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN"};
	Ptr<Tracker> tracker;
	_objectTracking(){
		tracker = TrackerKCF::create();
	}

	void init(Mat frame,Rect2d box){
		tracker->init(frame, box);
	}

	bool update(Mat frame, Rect2d &box){
		return tracker->update(frame, box);
	}
};

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
		qDebug()<<"Clearing objects.";
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
#if 1
		double lastMax = DBL_MAX;
		if(getObjectCount() > 0){
			int oi = 0;
			double minDisp;
			QHashIterator<int, QList<Point> > i(objects);
			while(i.hasNext()){
				i.next();

				int dispt = compareImage(objectsLastMat.value(i.key()), mt);

				if(dispt < lastMax){
					oi = i.key();
					minDisp = dispt;
				}
			}

			if(minDisp > CLOSE_VALUE){
				addObject();
				addObjectPoint(lastkey, mp, mt);
				oi = lastkey;
				qDebug()<<"add object: "<< minDisp << "oi: " << oi;
			}
			qDebug()<< "min disp: " << minDisp;
			return oi;
		}
		return -1;
#elif 0
		if(getObjectCount() > 0){
			int lastMax = 0;
			int oi = 0;

			QHashIterator<int, QList<Point> > i(objects);
			while(i.hasNext()){
				i.next();
				if(!objectsLastMat.contains(i.key()))
					break;

				int matchCount = compareImage(objectsLastMat.value(i.key()), mt);

				if(matchCount > lastMax){
					oi = i.key();
					lastMax = matchCount;
				}
			}
			if(lastMax < CLOSE_VALUE){
				addObject();
				addObjectPoint(lastkey, mp, mt);
				oi = lastkey;
			}
			return oi;
		}
		return -1;
#elif 0
		double lastMax = DBL_MAX;
		if(getObjectCount() > 0){
			int oi = 0;
			double minDisp;
			QHashIterator<int, QList<Point> > i(objects);
			while(i.hasNext()){
				i.next();
				Point p2(i.value().last().x, i.value().last().y);
				double dispt = norm(Mat(mp), Mat(p2));

				if(dispt < lastMax){
					oi = i.key();
					minDisp = dispt;
				}
			}
			if(minDisp > CLOSE_VALUE){
				addObject();
				addObjectPoint(lastkey, mp, mt);
				oi = lastkey;
			}
			//qDebug()<< "min disp: " << minDisp;
			return oi;
		}
#endif
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
			if(i.value().size() > 2)
				if(countRect.contains(i.value().first())){
					int p = isPosition(lineStart, lineEnd, i.value().last());
					int os = objectsPosState.value(i.key());
					//qDebug()<<QString("p: %1 s: %2").arg(p).arg(os);
					if(os == 2)
						objectsPosState.insert(i.key(), p);
					else if (os == 1 && p == -1){
						//up count
						objectUpCount++;
						objectsPosState.insert(i.key(), p);
					}else if (os == -1 && p == 1){
						//down count
						objectDownCount++;
						objectsPosState.insert(i.key(), p);
					}
				}
		}
	}

	int compareImage(Mat img_1, Mat img_2){

		Mat src_base, hsv_base;
		Mat src_test1, hsv_test1;

		src_base = img_1;
		src_test1 = img_2;

		/// Convert to HSV
		cvtColor( src_base, hsv_base, COLOR_BGR2HSV );
		cvtColor( src_test1, hsv_test1, COLOR_BGR2HSV );

		/// Using 50 bins for hue and 60 for saturation
		int h_bins = 50; int s_bins = 60;
		int histSize[] = { h_bins, s_bins };

		// hue varies from 0 to 179, saturation from 0 to 255
		float h_ranges[] = { 0, 180 };
		float s_ranges[] = { 0, 256 };

		const float* ranges[] = { h_ranges, s_ranges };

		// Use the o-th and 1-st channels
		int channels[] = { 0, 1 };


		/// Histograms
		MatND hist_base;
		MatND hist_test1;

		/// Calculate the histograms for the HSV images
		calcHist( &hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false );
		normalize( hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat() );

		calcHist( &hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false );
		normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

		double srate = 0;
		/// Apply the histogram comparison methods
		for( int i = 0; i < 4; i++ ){
			int compare_method = i;
			double base_base = compareHist( hist_base, hist_base, compare_method );
			double base_test1 = compareHist( hist_base, hist_test1, compare_method );

			srate += 100*abs(base_base-base_test1);
			//qDebug( " Method [%d] Perfect, Base-Half, Base-Test(1), Base-Test(2) : %f, %f", i, base_base, base_test1 );
		}
		//qDebug()<<"Done"<<(int) floor(srate);

		//imshow("debug1", img_1);
		//imshow("debug2", img_2);
		//mypause();
		return (int) floor(srate);
#if 0
		cvtColor(img_1, img_1, CV_BGR2GRAY);
		cvtColor(img_2, img_2, CV_BGR2GRAY);
		if( !img_1.data || !img_2.data )
		{ qDebug()<< " --(!) Error reading images "; return -1; }

		//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
		int minHessian = 400;
		Ptr<SURF> detector = SURF::create();
		detector->setHessianThreshold(minHessian);

		std::vector<KeyPoint> keypoints_1, keypoints_2;

		Mat descriptors_1, descriptors_2;
		detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
		detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

		//-- Step 2: Matching descriptor vectors using FLANN matcher

		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;

		if(descriptors_1.type()!=CV_32F) {
			descriptors_1.convertTo(descriptors_1, CV_32F);
		}
		if(descriptors_2.type()!=CV_32F) {
			descriptors_2.convertTo(descriptors_2, CV_32F);
		}

		matcher.match( descriptors_1, descriptors_2, matches );

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < descriptors_1.rows; i++ ){
			double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}
		printf("-- Max dist : %f \n", max_dist );
		printf("-- Min dist : %f \n", min_dist );
		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
		//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
		//-- small)
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ if( matches[i].distance <= max(2*min_dist, 0.02) )
			{ good_matches.push_back( matches[i]); }
		}
		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( img_1, keypoints_1, img_2, keypoints_2,
					 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		//-- Show detected matches
		imshow( "Good Matches", img_matches );
		for( int i = 0; i < (int)good_matches.size(); i++ )
		{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
		qDebug()<<good_matches.size();
		//mypause();
		return good_matches.size();
#endif
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

	if(!isFirst){
		minArea = (int) ((frame1.rows * frame1.cols)/2000);
		for(int i = 0; i < 100; i++)
			clr.append(Scalar(rand()%255, rand()%255, rand()%255));
	}

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
			if (area > minArea){
				contour_moments[i] = moments(contours[i], false);
				mass_centers[i] = Point(contour_moments[i].m10 / contour_moments[i].m00, contour_moments[i].m01 / contour_moments[i].m00);

				if(countRect.contains(mass_centers[i])){
				// Draw target
				Rect roi = boundingRect(contours[i]);
				drawContours(frame1, contours, i, Scalar(0, 0, 255));
				rectangle(frame1, roi, Scalar(0, 0, 255));
				drawTarget(mass_centers[i],frame1,i);

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
			createTrackbar("CLOSE_VALUE", "Movemont Detection", &c_slider, 50000, on_trackbar);
		}
		if(isCountmod)
			setMouseCallback("Movemont Detection", onmouse, &frame1);

		isFirst = true;
	}
}
