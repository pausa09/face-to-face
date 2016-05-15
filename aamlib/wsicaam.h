#ifndef WSICAAM_H
#define WSICAAM_H

#include "aam.h"

class WSICAAM : public AAM
{
private:
	cv::Mat calcWeights();
public:
    WSICAAM();

    void train();
    float fit();
};

#endif // WSICAAM_H
