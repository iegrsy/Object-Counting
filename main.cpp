#include <QApplication>
#include "objectcounter.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	ObjectCounter oc;
	VideoCapture capture("../video.avi");
	Mat frame;
	if( capture.isOpened() ){
		while( true ){
			capture.read(frame);
			if( !frame.empty() ){
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

