#include "FaceTracking.h"
#include "ChartWidget.h"
#include "MimRec.h"
#include "PrisonersDilemma.h"
#include "BackProp.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <QDirIterator>
#include <QProgressBar>

#include <opencv2/opencv.hpp>
#include <QtDebug>
#include <QDir>
#include <QApplication>
#include <QtCore/QTimer>
#include <QtNetwork/QTcpSocket>

#include <../AAMlib/icaam.h>
#include <../AAMlib/trainingdata.h>

const auto WINDOW_NAME = "MimRec";
static const int numParameters = 25;          //Number of used shape parameters
static const float fitThreshold = 0.05f;      //Termination condition

FaceTracking::FaceTracking()
{
	
	QObject::connect(&theServer, &QTcpServer::newConnection, this, &FaceTracking::newConnection);
	qDebug() << "Listening for Unity...";
	
	theServer.listen(QHostAddress::LocalHost, 12345);

	TrainingData neutralFace;
	//neutral is need for comparison
	neutralFace.loadDataFromFile("C:\\Development\\Test_fexrec\\train\\s077\\neutral.xml");
	                          
	calculator = new MimRec(neutralFace);

	pd = new PrisonersDilemma(calculator);

	TrainingData t;
	t.loadDataFromFile("C:\\Development\\Test_fexrec\\candide.xml");
	descriptions = t.getDescriptions();
	groups = t.getGroups();
	cv::Mat points = t.getPoints();
	QDirIterator dirIt("C:\\Development\\Test_fexrec\\train\\s077\\",
						{ "*.xml" }, QDir::Files, QDirIterator::Subdirectories);

	std::cout << " Loading data.." << std::endl;

	while (dirIt.hasNext())
	{
		std::cout << ".";

		dirIt.next();

		loadTrainingDataNew(dirIt.filePath().toStdString());
	};

	//Train aam with Training Data
	//optional: Set number of used Shape/Appearance Parameters
	aam.setNumShapeParameters(numParameters);
	aam.setNumAppParameters(numParameters);
	aam.train();

	cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

	pStaticCalcChartWidget = new ChartWidget(calculator, true);
	pStaticCalcChartWidget->show();

	pBackPropChartWidget = new ChartWidget(calculator, false);
	pBackPropChartWidget->show();

	QTimer::singleShot(0, this, &FaceTracking::run);
}


void FaceTracking::newConnection()
{
	if (theServer.hasPendingConnections())
	{
		qDebug() << "New connection!";

		if (theSocket != nullptr)
		{
			delete theSocket;
		}

		theSocket = theServer.nextPendingConnection();
		QObject::connect(theSocket, &QTcpSocket::disconnected, [this]{ theSocket->deleteLater(); theSocket = nullptr; });
		QObject::connect(theSocket, &QTcpSocket::readyRead, this, [this](){ processDataFromSocket(); });
	}
	else
	{
		Q_ASSERT(false);
	}
}
void FaceTracking::processDataFromSocket(bool forceSend)
{
	if (theSocket != nullptr)
	{
		if (forceSend || (!theSocket->bytesAvailable() != 0))
		{
			//We do not acutally care for what the client has send
			theSocket->readAll();

			const auto& vec = pd->getCurrentAgentAction();

			theSocket->putChar(static_cast<char>(vec.size()));
			theSocket->write(vec.data(), vec.size());
			theSocket->flush();
		}
	}
}

void FaceTracking::loadTrainingDataNew(const string &fileName)
{
	TrainingData t;
	t.loadDataFromFile(fileName);

	cv::Mat p = t.getPoints();
	cv::Mat i = t.getImage();

	i.convertTo(i, CV_32FC3);
	cv::cvtColor(i, i, CV_BGR2GRAY);
	aam.addTrainingData(p, i);
}

void FaceTracking::drawShape(const cv::Mat& image, const cv::Mat& points)
{
	if (!aam.triangles.empty())
	{
		for (int i = 0; i<aam.triangles.rows; i++) {
			cv::Point a, b, c;
			a = aam.getPointFromMat(points, aam.triangles.at<int>(i, 0));
			b = aam.getPointFromMat(points, aam.triangles.at<int>(i, 1));
			c = aam.getPointFromMat(points, aam.triangles.at<int>(i, 2));

			line(image, a, b, cv::Scalar(255, 0, 255), 1);
			line(image, a, c, cv::Scalar(255, 0, 255), 1);
			line(image, b, c, cv::Scalar(255, 0, 255), 1);
		}
	}

	cv::imshow(WINDOW_NAME, image);
}

#include <QMessageBox>

void FaceTracking::run()
{

	cv::Mat fittingImage;
	int i = 0;
	QDirIterator dirIm("C:\\Development\\Test_fexrec\\images\\s077\\",
	{ "*.png", "*.jpg" }, QDir::Files, QDirIterator::Subdirectories);

	while (dirIm.hasNext())
	{
		dirIm.next();

		fittingImage = cv::imread(dirIm.filePath().toStdString());

		/* this can be used for video capture
		cv::VideoCapture cap(0); // open the default camera
		if (!cap.isOpened())  // check if we succeeded
		{
			std::cout << "detection failed" << std::endl;
		}

		cv::Mat imageFromCap;

		imageFromCap.create(480, 640, CV_8UC3);

		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		frame.copyTo(imageFromCap);

		cap.release();

		*/
		//fittingImage = imageFromCap;
		i++;
		cv::Mat image = fittingImage.clone();
		aam.setFittingImage(fittingImage);   //Converts image to right format
		aam.resetShape();					 //Uses Viola-Jones Face Detection to initialize shape
	
		
		//Initialize with value > fitThreshold to enter the fitting loop
		float fittingChange = 20.0f;
		int steps = 0;

		while (fittingChange > fitThreshold && steps < 100)
		{
			fittingChange = aam.fit();		//Execute single update step
			steps++;
			//cout << "Step " << steps << " || Error per pixel: " << aam.getErrorPerPixel() << " || Parameter change: " << fittingChange << endl;
			cv::Mat image = fittingImage.clone();
			cv::Mat p = aam.getFittingShape();
			drawShape(image, p);
		}

		//Draw the final triangulation and display the result
		cv::Mat p = aam.getFittingShape();

		drawShape(image, p);

		//Save the result
		TrainingData tr;
		tr.setImage(fittingImage);
		tr.setPoints(p);
		tr.setGroups(groups);
		tr.setDescriptions(descriptions);
		for (int i_save = 0; i_save < i; ++i_save)
		{
			tr.saveDataToFile("out" + to_string(i) + ".xml");
		}
	

		calculator->calcForTrainingData(p,
			QDir::toNativeSeparators( dirIm.fileInfo().dir().absoluteFilePath( QStringLiteral( "facs.csv" ) ) ).toStdString() , fittingImage);
		
		
		//data were send to Chart Widget to display the intensity score
		pStaticCalcChartWidget->emitDataChanged();
		pBackPropChartWidget->emitDataChanged();

		processDataFromSocket(true);

		QApplication::processEvents();
		
		//starts the prisoners Dilemma
		pd->beginGame();

		//QMessageBox::information(nullptr, "Action Unit Detection", "Do you want to send these Action Units?");
	}
}

FaceTracking::~FaceTracking()
{
	delete calculator;
}
