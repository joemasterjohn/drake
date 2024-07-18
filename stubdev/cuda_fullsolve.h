#include <stdio.h>

#include <iostream>

#include "cuda_gpu_collision.h"

void FullSolveSapCPUEntry(Sphere* h_spheres, const int numProblems,
                          const int numSpheres, const int numContacts);