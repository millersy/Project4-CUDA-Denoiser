#pragma once

#include <vector>
#include "scene.h"
#include "timer.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void showDenoise(uchar4* pbo, int iter, int filterSize);
void denoiseIteration(int index, int x, int y, int step, glm::ivec2 resolution, GBufferPixel* gBuffer);
void pingPongGbuffer(int index, GBufferPixel* gBuffer);

PerformanceTimer& timer();
