#ifndef ICAAM_H
#define ICAAM_H

#include <opencv2/opencv.hpp>
#include "aam.h"


//Inverse Compositional AAM
class ICAAM: public AAM
{
private:
    void projectOutAppearanceVariation();

public:
    ICAAM();

    cv::Mat R;  //Inverse Hessian * Transposed SteepestDescentImages

    void train();

    //From aamfit
    int steps_global;
    int steps_shape;

    float fit();
};

#endif // AAM_H
