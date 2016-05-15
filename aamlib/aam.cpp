#include "aam.h"

#define fl at<float>


AAM::AAM()
{
    this->numPoints = 0;
    this->numAppParameters = 0;
    this->numShapeParameters = 0;
    this->initialized = false;
    this->steps = 0;
}

void AAM::addTrainingData(const  cv::Mat &shape, const  cv::Mat &image) {
	cv::Mat s = shape.reshape(1, 1).t();
	cv::Mat i = image.clone();

    switch(i.type()) {
    case CV_8UC3:
        i.convertTo(i, CV_32FC3);
        cvtColor(i,i,CV_BGR2GRAY);
        break;
    }

    //Mat i = image.reshape(1,1).t();

    if(this->trainingShapes.cols == 0) {
        this->trainingShapes = s;
        this->trainingImages.push_back(i);
    } else {
        hconcat(this->trainingShapes, s, this->trainingShapes);
        this->trainingImages.push_back(i);
    }
}

void AAM::calcShapeData() {
	cv::Mat pcaShapeData = this->trainingShapes.clone();
	cv::Mat mean = AAM::alignShapeData(pcaShapeData);

	cv::PCA shapePCA = cv::PCA(pcaShapeData,
						cv::Mat(),
                       CV_PCA_DATA_AS_COL
                       );

    pcaShapeData.release();

    this->s0 = mean;
    this->numPoints = s0.rows/2;
    this->s = shapePCA.eigenvectors;

    if(this->numShapeParameters > 0 && this->numShapeParameters <= this->s.rows) {
        this->s = this->s(cv::Rect(0,0,this->s.cols, this->numShapeParameters));
    }

	this->s_star = cv::Mat::zeros(4, 2 * this->numPoints, CV_32FC1);

    for(int i=0; i<this->numPoints; i++) {
        s_star.fl(0, 2*i) = s0.fl(2*i);
        s_star.fl(0, 2*i+1) = s0.fl(2*i+1);
        s_star.fl(1, 2*i) = -s0.fl(2*i+1);
        s_star.fl(1, 2*i+1) = s0.fl(2*i);
        s_star.fl(2, 2*i) = 1;
        s_star.fl(2, 2*i+1) = 0;
        s_star.fl(3, 2*i) = 0;
        s_star.fl(3, 2*i+1) = 1;
    }

    this->calcTriangleStructure(this->s0);
	this->p = cv::Mat::zeros(this->s.rows + this->s_star.rows, 1, CV_32FC1);
}

void AAM::calcAppearanceData() {
    int inputSize = this->trainingShapes.cols;
	cv::Mat pcaAppearanceData = cv::Mat(this->modelWidth*this->modelHeight, inputSize, CV_32FC1);

    for(int i=0; i<inputSize; i++) {
		cv::Mat image = trainingImages.at(i).clone();

        image = AAM::warpImageToModel(image, trainingShapes.col(i));

		normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32FC1); //TODO: use mask to normalize only over values inside shape

        image = image.reshape(1,1).t();

        image.copyTo(pcaAppearanceData.col(i));
        image.release();
    }

	cv::PCA appearancePCA = cv::PCA(pcaAppearanceData,
						cv::Mat(),
                        CV_PCA_DATA_AS_COL
                        );

    this->A0 = appearancePCA.mean;
    //normalize(this->A0, this->A0, 0, 1, NORM_MINMAX, CV_32FC1);
    this->A = appearancePCA.eigenvectors;

    if(this->numAppParameters > 0 && this->numAppParameters <= this->A.rows) {
        this->A = this->A(cv::Rect(0,0,this->A.cols, this->numAppParameters));
    }

	this->lambda = cv::Mat::zeros(this->A.rows, 1, CV_32FC1);
}

void AAM::calcGradients() {
    this->calcGradX();
    this->calcGradY();
}

void AAM::calcGradX() {
	gradX = cv::Mat::zeros(modelHeight, modelWidth, CV_32FC1);

    Sobel( this->A0.reshape(1,this->modelHeight), gradX, CV_32FC1, 1, 0, 1);
    gradX = gradX.reshape(1,1);

	gradXA = cv::Mat::zeros(this->A.rows, this->A.cols, CV_32FC1);
    for(int i=0; i<this->A.rows; i++) {
		cv::Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, CV_32FC1, 1, 0, 1);

        gradXA.row(i) = grad.reshape(1,1);
    }
}

void AAM::calcGradY() {
	gradY = cv::Mat::zeros(modelHeight, modelWidth, CV_32FC1);

    Sobel( this->A0.reshape(1, this->modelHeight), gradY, CV_32FC1, 0, 1, 1);
    gradY = gradY.reshape(1,1);

	gradYA = cv::Mat::zeros(this->A.rows, this->A.cols, CV_32FC1);
    for(int i=0; i<this->A.rows; i++) {
		cv::Mat grad;
        Sobel(this->A.row(i).reshape(1,this->modelHeight), grad, CV_32FC1, 0, 1, 1);

        gradYA.row(i) = grad.reshape(1,1);
    }
}

void AAM::calcSteepestDescentImages() {
	cv::Mat sdImg;

	cv::Mat X = gradX + (this->gradXA.t()*this->lambda).t();
	cv::Mat Y = gradY + (this->gradYA.t()*this->lambda).t();

    for(int i=0; i<this->jacobians.rows/2; i++) {
		cv::Mat descentImage = X.mul(this->jacobians.row(2 * i)) + Y.mul(this->jacobians.row(2 * i + 1));
        sdImg.push_back(descentImage);
    }

    this->steepestDescentImages = sdImg;
}

void AAM::calcJacobian() {
	cv::Mat j = cv::Mat::zeros(2 * s.rows + 2 * s_star.rows, this->A.cols, CV_32FC1);

    for(int i=0; i<numPoints; i++) {
		cv::Mat derivate = this->derivateWarpToPoint(i); //dw/dx
        derivate = derivate.reshape(1,1);

        for(int globalTrans=0; globalTrans<this->s_star.rows; globalTrans++) {
            j.row(2*globalTrans) += derivate*this->s_star.fl(globalTrans, 2*i);
            j.row(2*globalTrans+1) += derivate*this->s_star.fl(globalTrans,  2*i+1);
        }

        for(int shapeVector=0; shapeVector < this->s.rows; shapeVector++) {
            j.row(2*shapeVector+2*this->s_star.rows) += derivate*this->s.fl(shapeVector, 2*i); //x-component
            j.row(2*shapeVector+2*this->s_star.rows+1) += derivate*this->s.fl(shapeVector, 2*i+1); //y-component
        }
    }

    this->jacobians = j;
}

cv::Mat AAM::derivateWarpToPoint(int vertexId) {
	cv::Mat derivate = cv::Mat::zeros(modelHeight, modelWidth, CV_32F);
    std::vector<int> tris = this->triangleLookup[vertexId];
    int numTriangles = tris.size();
    for(int j=0; j<numTriangles; j++) {
        int a,b,c;
        int triId = tris[j];
        a = this->triangles.at<int>(triId,0);
        b = this->triangles.at<int>(triId,1);
        c = this->triangles.at<int>(triId,2);

        this->setFirstPoint(vertexId, a, b, c);

		cv::Point2f pa, pb, pc;
        pa = AAM::getPointFromMat(s0, a);
        pb = AAM::getPointFromMat(s0, b);
        pc = AAM::getPointFromMat(s0, c);

        //Bereich des Dreiecks berechnen um nicht ganzes Bild zu durchsuchen
		int min_x = floor(cv::min(pa.x, cv::min(pb.x, pc.x)));
		int max_x = ceil(cv::max(pa.x, cv::max(pb.x, pc.x)));
		int min_y = floor(cv::min(pa.y, cv::min(pb.y, pc.y)));
		int max_y = ceil(cv::max(pa.y, cv::max(pb.y, pc.y)));

        for(int row=min_y; row<max_y; row++) {
            for(int col=min_x; col<max_x; col++) {
				cv::Point px;
                px.x = col;
                px.y = row;

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                if((alpha >= 0) && (beta >= 0) && (alpha + beta <= 1)) {
                    //Nur wenn Punkt innerhalb des Dreiecks liegt
                    float val = 1-alpha-beta;

                    if(val > 0) {
                        derivate.fl(row,col) = val;
                    }
                }
            }
        }
    }

    return derivate;
}


cv::Point2f AAM::calcMean(const  cv::Mat &m) {
    int n = m.rows;
	cv::Point2f mean = cv::Point2f(0, 0);

    for(int j = 0; j < n; j++) {
        mean.x += m.fl(j,0);
        mean.y += m.fl(j,1);
    }

    mean.x /= n;
    mean.y /= n;

    return mean;
}

cv::Mat AAM::moveToOrigin(const  cv::Mat &m) {
	cv::Mat Out = m.clone();
    int n = Out.rows;

	cv::Point2f mean = this->calcMean(Out);

    for(int j = 0; j < n; j++) {
        Out.fl(j,0) = Out.fl(j,0) - mean.x;
        Out.fl(j,1) = Out.fl(j,1) - mean.y;
    }

    return Out;
}

cv::Mat AAM::procrustes(const cv::Mat &X, const  cv::Mat &Y) {
    int n = X.rows/2;

    //Reshape to have col(0)=x and col(1)=y
	cv::Mat X0 = X.reshape(1, n);
	cv::Mat Y0 = Y.reshape(1, n);

    //move center to (0,0)
    X0 = this->moveToOrigin(X0);
    Y0 = this->moveToOrigin(Y0);

    float normX = sqrt(sum(X0.mul(X0))[0]);
    float normY = sqrt(sum(Y0.mul(Y0))[0]);

    X0 /= normX;
    Y0 /= normY;

	cv::Mat U, Vt, D;
	cv::Mat M = X0.t()*Y0;
	cv::SVD::compute(M, D, U, Vt);

    //Rotation
	cv::Mat R = Vt.t()*U.t();

	cv::Mat x = Y0.col(0).clone();
	cv::Mat y = Y0.col(1).clone();

    Y0.col(0) = R.fl(0,0)*x + R.fl(1,0)*y;
    Y0.col(1) = R.fl(0,1)*x + R.fl(1,1)*y;

    //Scaling
    float scaling = sum(D)[0];

	cv::Mat Out = normX*scaling*Y0;


	cv::Point2f meanX = calcMean(X.reshape(1, n));

    for(int i=0; i<n; i++) {
        Out.fl(i,0) += meanX.x;
        Out.fl(i,1) += meanX.y;
    }

    Out = Out.reshape(1,2*n);

    return Out;
}

//returns mean shape
cv::Mat AAM::alignShapeData(cv::Mat &shapeData) {
	cv::Mat S = shapeData.clone();
    int numPoints = S.rows;
    int numShapes = S.cols;

	cv::Mat meanShape = S.col(0).clone();
	cv::Mat referenceShape = S.col(0).clone();

    float meanChange = 20.0f;

    while(meanChange > 0.1) {
        for(int i=0; i<numShapes; i++) {
			cv::Mat Y = procrustes(meanShape.clone(), S.col(i).clone());
            Y.copyTo(S.col(i));
        }

		cv::Mat newMeanShape = cv::Mat::zeros(numPoints, 1, CV_32FC1);

        for(int i=0; i<numPoints; i++) {
            float meanVal = 0;
            for(int j=0; j<numShapes; j++) {
                meanVal += S.fl(i,j);
            }
            meanVal /= numShapes;
			// newMeanShape.fl(0,i) = meanVal;
			newMeanShape.fl(i, 0) = meanVal;
          
        }

		cv::Mat Y = procrustes(referenceShape, newMeanShape);
        newMeanShape = Y;

        meanChange = sum(abs(meanShape-newMeanShape))[0];

        meanShape = newMeanShape;
        newMeanShape.release();
    }

	cv::Mat tempShape = meanShape.clone().reshape(1, numPoints / 2);

    double minX,maxX,minY,maxY;

	cv::minMaxIdx(tempShape.col(0), &minX, &maxX);
    minMaxIdx(tempShape.col(1), &minY, &maxY);

	tempShape = tempShape - repeat((cv::Mat_<float>(1, 2) << minX - 2, minY - 2), numPoints / 2, 1);

    this->modelWidth = ceil(maxX-minX)+4;
    this->modelHeight = ceil(maxY-minY)+4;

    meanShape = tempShape.reshape(1, numPoints);

    S = S - repeat(meanShape,1,numShapes);

    shapeData = S;
    return meanShape;
}

void AAM::calcTriangleStructure(const  cv::Mat &s) {
	std::vector< cv::Vec6f> triangleList;
    bool insert;
	cv::Subdiv2D subdiv;
    int counter = 0;

	subdiv.initDelaunay(cv::Rect(0, 0, this->modelWidth, this->modelHeight));

    for(int i=0; i<this->numPoints; i++) {
		cv::Point2f v = this->getPointFromMat(s, i);
        subdiv.insert(v);
    }

    subdiv.getTriangleList(triangleList);

    triangleLookup.clear();

    for(int i=0; i<this->numPoints; i++) {
        std::vector <int> temp;
        this->triangleLookup.push_back(temp);
    }

	this->triangles = cv::Mat(triangleList.size(), 3, CV_32S);

    for(unsigned int i=0; i<triangleList.size(); i++) {
        cv::Vec6f t = triangleList[i];
        std::vector<cv::Point2f> pt(3);

		pt[0] = cv::Point2f(t[0], t[1]);
		pt[1] = cv::Point2f(t[2], t[3]);
		pt[2] = cv::Point2f(t[4], t[5]);

        insert = true;

        for(int j=0; j<3; j++) {
            if(pt[j].x > modelWidth || pt[j].y > modelHeight || pt[j].x < 0 || pt[j].y < 0) {
                insert = false;
                break;
            }
        }

        if(insert) {
            if(pt[0]!=pt[1] && pt[0]!=pt[2] && pt[1]!=pt[2]) {
                int posA, posB, posC;
                posA = this->findPointInShape(pt[0]);
                posB = this->findPointInShape(pt[1]);
                posC = this->findPointInShape(pt[2]);

                this->triangles.at<int>(counter, 0) = posA;
                this->triangles.at<int>(counter, 1) = posB;
                this->triangles.at<int>(counter, 2) = posC;

                this->triangleLookup[posA].push_back(counter);
                this->triangleLookup[posB].push_back(counter);
                this->triangleLookup[posC].push_back(counter);

                counter++;
            }
        }
    }

    this->triangles = this->triangles(cv::Rect(0,0,3,counter));
}

int AAM::findPointInShape(const  cv::Point2f &p) {
    for(int i=0; i<this->numPoints; i++) {
		cv::Point2f s = getPointFromMat(this->s0, i);
        if(s == p) {
            return i;
        }
    }

    return -1;
}

cv::Point2f AAM::getPointFromMat(const  cv::Mat &m, int pointId) {
	return  cv::Point2f(m.fl(2 * pointId), m.fl(2 * pointId + 1));
}

cv::Mat AAM::warpImageToModel(const  cv::Mat &inputImage, const  cv::Mat &inputPoints) {
	cv::Mat out = cv::Mat::zeros(modelHeight, modelWidth, inputImage.type());
    return warpImage(inputImage, inputPoints, out, this->s0);
}

cv::Mat AAM::warpImage(const  cv::Mat &inputImage, const  cv::Mat &inputPoints, const  cv::Mat &outputImage, const  cv::Mat &outputPoints) {
	cv::Mat warpedImage = outputImage;
    int triSize = this->triangles.rows;

    for(int i=0; i<triSize; i++) {
		cv::Point2f srcTri[3], dstTri[3];
        int a,b,c;
        a = this->triangles.at<int>(i,0);
        b = this->triangles.at<int>(i,1);
        c = this->triangles.at<int>(i,2);

        srcTri[0] = this->getPointFromMat(inputPoints, a);
        srcTri[1] = this->getPointFromMat(inputPoints, b);
        srcTri[2] = this->getPointFromMat(inputPoints, c);

        dstTri[0] = this->getPointFromMat(outputPoints, a);
        dstTri[1] = this->getPointFromMat(outputPoints, b);
        dstTri[2] = this->getPointFromMat(outputPoints, c);

        warpTextureFromTriangle(srcTri, inputImage, dstTri, warpedImage);
    }

    return warpedImage;
}

void AAM::warpTextureFromTriangle(cv::Point2f srcTri[], const  cv::Mat &originalImage, cv::Point2f dstTri[], cv::Mat warp_final) {
	cv::Mat warp_mat(2, 3, CV_32FC1);
	cv::Mat warp_dst, warp_mask, srcImg;
    int smoothingParam = 1;

	int min_x_src = floor(cv::min(srcTri[0].x, cv::min(srcTri[1].x, srcTri[2].x)));
	int max_x_src = ceil(cv::max(srcTri[0].x, cv::max(srcTri[1].x, srcTri[2].x)));
	int min_y_src = floor(cv::min(srcTri[0].y, cv::min(srcTri[1].y, srcTri[2].y)));
	int max_y_src = ceil(cv::max(srcTri[0].y, cv::max(srcTri[1].y, srcTri[2].y)));

	int src_size_x = cv::max(max_x_src - min_x_src, 1) + 2 * smoothingParam;
	int src_size_y = cv::max(max_y_src - min_y_src, 1) + 2 * smoothingParam;

    srcImg = originalImage(cv::Rect_<int>(min_x_src-smoothingParam,min_y_src-smoothingParam,src_size_x,src_size_y));
    for(int i=0; i<3; i++) {
		srcTri[i] -= cv::Point2f(min_x_src - smoothingParam, min_y_src - smoothingParam);
    }

	int min_x_dst = floor(cv::min(dstTri[0].x, cv::min(dstTri[1].x, dstTri[2].x)));
	int max_x_dst = ceil(cv::max(dstTri[0].x, cv::max(dstTri[1].x, dstTri[2].x)));
	int min_y_dst = floor(cv::min(dstTri[0].y, cv::min(dstTri[1].y, dstTri[2].y)));
	int max_y_dst = ceil(cv::max(dstTri[0].y, cv::max(dstTri[1].y, dstTri[2].y)));

    int dst_size_x = max_x_dst - min_x_dst;
    int dst_size_y = max_y_dst - min_y_dst;

    for(int i=0; i<3; i++) {
		dstTri[i] -= cv::Point2f(min_x_dst, min_y_dst);
    }

	cv::Point triPoints[3];
    triPoints[0] = dstTri[0];
    triPoints[1] = dstTri[1];
    triPoints[2] = dstTri[2];

	warp_dst = cv::Mat::zeros(dst_size_y, dst_size_x, originalImage.type());
	warp_mask = cv::Mat::zeros(dst_size_y, dst_size_x, CV_8U);

    // Get the Affine Transform
    warp_mat = getAffineTransform(srcTri, dstTri);

    // Apply the Affine Transform to the src image
    warpAffine(srcImg, warp_dst, warp_mat, warp_dst.size());
	fillConvexPoly(warp_mask, triPoints, 3, cv::Scalar(255, 255, 255), CV_AA, 0);
    warp_dst.copyTo(warp_final(cv::Rect_<int>(min_x_dst, min_y_dst, dst_size_x, dst_size_y)), warp_mask);
}

void AAM::setNumShapeParameters(int num) {
    this->numShapeParameters = num;
}

void AAM::setNumAppParameters(int num) {
    this->numAppParameters = num;
}

void AAM::setFittingImage(const cv::Mat &fittingImage) {
    fittingImage.convertTo(this->fittingImage, CV_32FC3);
    cvtColor(this->fittingImage, this->fittingImage, CV_BGR2GRAY);
	normalize(this->fittingImage, this->fittingImage, 0, 1, cv::NORM_MINMAX, CV_32FC1);
}

void AAM::setStartingShape(const  cv::Mat &shape) {
    this->fittingShape = shape;
}

void AAM::resetShape() {
	cv::CascadeClassifier faceCascade;
	std::vector< cv::Rect> faces;
	cv::Mat detectImage;

    this->fittingImage.convertTo(detectImage, CV_8UC1, 255);

    //faceCascade.load("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml");
	faceCascade.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml");
	faceCascade.detectMultiScale(detectImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(100, 100));

    if(faces.size() > 0) {
		cv::Mat fitPoints = this->s0.clone();

		cv::Rect face = faces[0];

        fitPoints = fitPoints.reshape(1, this->numPoints);
		cv::Mat x = fitPoints.col(0);
		cv::Mat y = fitPoints.col(1);

        double minX, minY, maxX, maxY;
        minMaxIdx(x, &minX, &maxX);
        minMaxIdx(y, &minY, &maxY);

        double scale = face.width/(maxX-minX);
        scale *= 0.8; //For best initial fit, depends on the used model

        //Center at (0,0)
        x = x-minX-(maxX-minX)/2;
        y = y-minY-(maxY-minY)/2;

        //Scale
        x *= scale;
        y *= scale;

        //Move to Position of found Face;
        x += face.x+face.width/2;
        y += face.y+face.height/2;

        this->fittingShape = fitPoints.reshape(1, this->numPoints*2);
    } else {
		cv::Rect face = cv::Rect(this->fittingImage.cols / 2 - 100, this->fittingImage.rows / 2 - 100, 200, 200);

		cv::Mat fitPoints = this->s0.clone();

        fitPoints = fitPoints.reshape(1, this->numPoints);
		cv::Mat x = fitPoints.col(0);
		cv::Mat y = fitPoints.col(1);

        double minX, minY, maxX, maxY;
        minMaxIdx(x, &minX, &maxX);
        minMaxIdx(y, &minY, &maxY);

        double scale = face.width/(maxX-minX);
        scale *= 0.8; //For best initial fit, depends on the used model

        //Center at (0,0)
        x = x-minX-(maxX-minX)/2;
        y = y-minY-(maxY-minY)/2;

        //Scale
        x *= scale;
        y *= scale;

        //Move to Position of found Face;
        x += face.x+face.width/2;
        y += face.y+face.height/2;

        this->fittingShape = fitPoints.reshape(1, this->numPoints*2);
    }

    this->steps = 0;
}

double AAM::getErrorPerPixel() {
    //return sum(abs(this->errorImage))[0]/(this->errorImage.rows*this->errorImage.cols);
    return sum(abs(this->errorImage))[0]/(countNonZero(this->errorImage));
}

cv::Mat AAM::getErrorImage() {
    return this->errorImage;
}

cv::Mat AAM::getFittingShape() {
    return this->fittingShape;
}

void AAM::setFirstPoint(int id, int &a, int &b, int &c) {
    if(a != id) {
        int temp = a;
        if(b == id) {
            a = b;
            b = temp;
        } else if(c == id) {
            a = c;
            c = temp;
        } else {
            std::cout<<"Error: Point not in Triangle "<<id<<" "<<a<<" "<<b<<" "<<c<< std::endl;
        }
    }
}

cv::Mat AAM::getAppearanceReconstruction() {
	cv::Mat appVar = this->A0 + this->A.t()*this->lambda;

    return appVar.reshape(1, this->modelHeight);
}

cv::Mat AAM::getAppearanceReconstructionOnFittingImage() {
	cv::Mat app = this->getAppearanceReconstruction();
	cv::Mat out = this->fittingImage.clone();
    return this->warpImage(app, this->s0, out, this->fittingShape);
}

void AAM::updateAppearanceParameters(cv::Mat deltaLambda) {
    this->lambda += deltaLambda;
}

void AAM::updateInverseWarp(cv::Mat deltaShapeParam) {
	cv::Mat deltaq = deltaShapeParam(cv::Rect(0, 0, 1, 4));
	cv::Mat deltap = deltaShapeParam(cv::Rect(0, 4, 1, deltaShapeParam.rows - 4));

    //Inverse Global Shape Transformation
	cv::Mat A(2, 2, CV_32FC1);
	cv::Mat t(1, 2, CV_32FC1);

    A.fl(0,0) = 1/(1 + deltaq.fl(0));
    A.fl(1,0) = deltaq.fl(1);
    A.fl(0,1) = -deltaq.fl(1);
    A.fl(1,1) = 1/(1 + deltaq.fl(0));

    t.fl(0,0) = -deltaq.fl(2);
    t.fl(0,1) = -deltaq.fl(3);

    /*
    deltaq.fl(2) = deltaq.fl(2)/this->fittingImage.cols;
    deltaq.fl(3) = deltaq.fl(3)/this->fittingImage.rows;
    */
    float warpChange = sum(abs(deltaq))[0];

   // cout<<warpChange<<endl;

    //Lower Values improve the fitting stability, but increase the runtime
    if(warpChange > 0.3) {
		cv::Mat deltaS0 = this->s0.clone().reshape(1, this->numPoints);

		cv::Mat Rot = deltaS0*A;
		cv::Mat tr = repeat(t, this->numPoints, 1);
        deltaS0 = Rot + tr - deltaS0;

        deltaS0 = deltaS0.reshape(1,2*this->numPoints);

        this->fittingShape += deltaS0;
    } else {
		cv::Mat deltaS0 = this->s0 - (deltap.t()*this->s).t();
        deltaS0 = deltaS0.reshape(1, this->numPoints);

		cv::Mat Rot = deltaS0*A;
		cv::Mat tr = repeat(t, this->numPoints, 1);
        deltaS0 = Rot + tr;

        deltaS0 = deltaS0.reshape(1,1).t();

        for(int i=0; i<this->numPoints; i++) {
			cv::Point2f update = cv::Point2f(0, 0);

			cv::Point2f px = AAM::getPointFromMat(deltaS0, i);
            std::vector<int> triangles = this->triangleLookup[i];

            int numTriangles = triangles.size();
            for(int j=0; j<numTriangles; j++) {
                int a,b,c;
                int triId = triangles[j];
                a = this->triangles.at<int>(triId,0);
                b = this->triangles.at<int>(triId,1);
                c = this->triangles.at<int>(triId,2);

                this->setFirstPoint(i, a, b, c);    // Sort points

				cv::Point2f pa, pb, pc;
                pa = AAM::getPointFromMat(this->s0, a);
                pb = AAM::getPointFromMat(this->s0, b);
                pc = AAM::getPointFromMat(this->s0, c);

                float den = (pb.x - pa.x)*(pc.y - pa.y)-(pb.y - pa.y)*(pc.x - pa.x);

                float alpha = ((px.x - pa.x)*(pc.y - pa.y)-(px.y - pa.y)*(pc.x - pa.x))/den;
                float beta = ((px.y - pa.y)*(pb.x - pa.x)-(px.x - pa.x)*(pb.y - pa.y))/den;

                pa = AAM::getPointFromMat(this->fittingShape, a);
                pb = AAM::getPointFromMat(this->fittingShape, b);
                pc = AAM::getPointFromMat(this->fittingShape, c);

                update += alpha*(pb-pa) + beta*(pc-pa);
            }

            update.x /= numTriangles;
            update.y /= numTriangles;

            this->fittingShape.fl(2*i) += update.x;
            this->fittingShape.fl(2*i+1) += update.y;
        }
    }
}

void AAM::calcErrorImage() {
	cv::Mat warpedImage = AAM::warpImageToModel(this->fittingImage, this->fittingShape);
	normalize(warpedImage, warpedImage, 0, 1, cv::NORM_MINMAX, CV_32FC1);

	cv::Mat errorImage = AAM::getAppearanceReconstruction() - warpedImage;
    this->errorImage = errorImage.reshape(1,1);
}

bool AAM::isInitialized() {
    return this->initialized;
}

bool AAM::hasFittingImage() {
    return !this->fittingImage.empty();
}
