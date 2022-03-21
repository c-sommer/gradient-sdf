/**
BSD 3-Clause License

This file is part of the code accompanying the paper
Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction
by Christiane Sommer*, Lu Sang*, David Schubert, and Daniel Cremers (* denotes equal contribution).

Copyright (c) 2021, Christiane Sommer and Lu Sang.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef TIMER_H_
#define TIMER_H_

// includes
#include <iostream>
#include <ctime>
#include <string>
#include <omp.h>

class Timer {

private:
    double start_time;
    double end_time;
    double elapsed;

public:
    Timer() : start_time(0.), end_time(0.), elapsed(0.) {}
    ~Timer() {}
    
    void tic() {
        start_time = omp_get_wtime();
    }
    
    double toc(std::string s = "Time elapsed") {
        if (start_time!=0) {
            end_time = omp_get_wtime();
            elapsed = end_time-start_time;
            print_time(s);
        }
        else
            std::cout << "Timer was not started, no time could be measured." << std::endl;
        return elapsed;
    }
    
    void print_time(std::string s = "Time elapsed") {        
        if (elapsed<1.)
            std::cout << "---------- " << s << ": " << 1000.*elapsed << "ms." << std::endl;
        else
            std::cout << "---------- " << s << ": " << elapsed << "s." << std::endl;
    }

};

#endif // TIMER_H_
