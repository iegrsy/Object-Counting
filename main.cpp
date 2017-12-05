#include <QApplication>
#include "objectcounter.h"
#include <QDebug>

class BouncingBalls
{
private:
	enum Directions{N,NO,O,SO,S,SW,W,NW};

	Directions dtn;
	Point location;
	int ball_size;
	int ball_speed;
	Scalar ball_color;

	void selectDirection(){
		int r = rand()%8+1;
		switch (r) {
		case 1:	dtn = N; break;
		case 2:	dtn = NO; break;
		case 3:	dtn = O; break;
		case 4:	dtn = SO; break;
		case 5:	dtn = S; break;
		case 6:	dtn = SW; break;
		case 7:	dtn = W; break;
		case 8:	dtn = NW; break;
		default: dtn = O; break;
		}
	}

	void updateBallLocation(Size s){
		Point location = this->location;
		switch (dtn) {
		case N:  location = Point(location.x, location.y - ball_speed); break;
		case NO: location = Point(location.x + ball_speed, location.y - ball_speed); break;
		case O:  location = Point(location.x + ball_speed, location.y); break;
		case SO: location = Point(location.x + ball_speed, location.y + ball_speed); break;
		case S:  location = Point(location.x, location.y + ball_speed); break;
		case SW: location = Point(location.x - ball_speed, location.y + ball_speed); break;
		case W:  location = Point(location.x - ball_speed , location.y); break;
		case NW: location = Point(location.x - ball_speed, location.y - ball_speed); break;
		}

		if(location.x < 0 || location.y < 0 ||
				location.x > s.width || location.y > s.height){
			selectDirection();
			updateBallLocation(s);
		}
		this->location = location;
		//qDebug()<<QString("Size:(%1, %2) Location:(%3, %4) Directions:%5").
		//		  arg(s.width).arg(s.height).arg(location.x).arg(location.y).arg(dtn);
	}

public:
	BouncingBalls() {
		selectDirection();
		location = Point(1,1);
		ball_size = (rand()%5+1) * 10;
		ball_speed = (rand()%5+1) * 5;
		ball_color = Scalar((rand()%255), (rand()%255), (rand()%255));
	}
	void updateBall(Mat &frame){
		updateBallLocation(frame.size());
		circle(frame, location, ball_size, ball_color,CV_FILLED, 8,0);
	}
};

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	ObjectCounter oc;
	bool isballtest = false;
#if 1
	VideoCapture capture("../../video.avi");
	/* test video
	 * ../../pirates-flashmob.h264
	 */
	//VideoCapture capture("../../pirates-flashmob.h264");
#else
	VideoCapture capture(0);
	isballtest = true;
#endif
	Mat frame;

	BouncingBalls bb;
	BouncingBalls bb1;
	BouncingBalls bb2;

	if( capture.isOpened() ){
		qDebug()<< QString("Video FPS: %1").arg(capture.get(CAP_PROP_FPS));
		while( true ){
			if(capture.read(frame)){
				if(isballtest){bb.updateBall(frame); bb1.updateBall(frame); bb2.updateBall(frame);}
				oc.movemontDetection(frame);
			}else{
				qDebug(" --(!) No captured frame -- Break!");
				break;
			}

			int c = waitKey(10);
			if( (char)c == 'q' ){
				break;
			}else if( (char)c == 'p' ){
				while (true) {
					int c = waitKey(10);
					if( (char)c == 'p' ){
						break;
					}
				}
			}
		}
		destroyAllWindows();
	}
	return 0;
	//return a.exec();
}

