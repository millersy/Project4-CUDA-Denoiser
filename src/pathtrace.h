#pragma once

#include <vector>
#include "scene.h"
#include "timer.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration, bool denoise, int filterSize, float c_weight, float p_weight, float n_weight);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoise(uchar4* pbo, int iter, int filterSize, float c_weight, float p_weight, float n_weight);

PerformanceTimer& timer();
