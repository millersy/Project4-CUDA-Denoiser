CUDA Denoiser For CUDA Path Tracer
==================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Sydney Miller
  * [LinkedIn](https://www.linkedin.com/in/sydney-miller-upenn/)
* Tested on: GTX 222 222MB (CETS Virtual Lab)

### README

| Original | Denoised |
| ------------- | ----------- | 
| ![](img/renders/prj4/noisyCornellBox.png)  | ![](img/renders/prj4/denoisedCornellBox.png)| 

### Results

#### Smoothness Based on Iterations 
| 1 Iteration | 3 Iterations | 7 Iterations | 10 Iterations |
| ------------- | ----------- | ------------- | ----------- |
| ![](img/renders/prj4/iterationTests/mirror-1.png)  | ![](img/renders/prj4/iterationTests/mirror-3.png)| ![](img/renders/prj4/iterationTests/mirror-7.png)  | ![](img/renders/prj4/iterationTests/mirror-10.png) |

#### Denoising Different Materials
| Specular | Diffuse | Refractive | 
| ------------- | ----------- | ------------- | 
| ![](img/renders/prj4/materialTests/mirror-10.png)  | ![](img/renders/prj4/materialTests/diffuse.png)| ![](img/renders/prj4/materialTests/glass.png) | 

#### Denoising With Different Filter Sizes
| 10 x 10 px | 25 x 25 px | 50 x 50 px | 100 x 100 px |
| ------------- | ----------- | ------------- | ----------- |
| ![](img/renders/prj4/filterTests/filter10.png)  | ![](img/renders/prj4/filterTests/filter25.png)| ![](img/renders/prj4/filterTests/filter50.png)  | ![](img/renders/prj4/filterTests/filter100.png) |

#### Comparing Different Scenes
| Large Light | Small Light |
| ------------- | ----------- | 
| ![](img/renders/prj4/denoisedCornellBox.png)  | ![](img/renders/prj4/denoisedCornellBetter.png)| 

### Performance Analysis

#### Runtime at Different Resolutions
![](img/renders/prj4/VaryingResolutionGraph.png)

#### Runtime with Varying Filter Sizes
![](img/renders/prj4/FilterSizeGraph.png)

