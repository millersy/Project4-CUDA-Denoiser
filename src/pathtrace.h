#pragma once

#include <vector>
#include "scene.h"
#include "timer.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoise(uchar4* pbo, int iter, int filterSize, int c_weight, int p_weight, int n_weight);
void denoiseIteration(int index, int x, int y, int step, int c_weight, int p_weight, int n_weight, glm::ivec2 resolution, GBufferPixel* gBuffer);
void pingPongGbuffer(int index, GBufferPixel* gBuffer);

PerformanceTimer& timer();
