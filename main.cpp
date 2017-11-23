#include <QCoreApplication>
#include "objectcounter.h"

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	ObjectCounter oc;
	VideoCapture capture(0);
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
			}
		}
		destroyAllWindows();
	}
	return 0;
	//return a.exec();
}

