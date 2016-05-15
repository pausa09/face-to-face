#ifndef SICAAM_H
#define SICAAM_H

#include "aam.h"

class SICAAM : public AAM
{
private:

public:
    SICAAM();

    void train();
    float fit();
};

#endif // SICAAM_H
