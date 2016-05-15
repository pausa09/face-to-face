#ifndef FASTSICAAM_H
#define FASTSICAAM_H

#include "aam.h"

class FastSICAAM : public AAM
{
public:
    FastSICAAM();

    void train();
    float fit();
};

#endif // FASTSICAAM_H
