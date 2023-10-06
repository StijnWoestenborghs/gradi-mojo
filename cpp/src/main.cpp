#include <iostream>
#include <Eigen/Dense>

#include "timer.h"
#include "gradient_descent.h"



int main(){
    std::cout << "[EXECUTABLE] Gradient Descent as C++ executable " << std::endl;

    const int N = 10;
    const int dim = 2;
    const float lr = 0.00001;
    const int niter = 1000;

    Eigen::Matrix<float, N, dim> X = Eigen::Matrix<float, N, dim>::Zero();
    Eigen::Matrix<float, N, N> D = Eigen::Matrix<float, N, N>::Zero();
    
    
    InputBindingInterface input;
    input.N = N;
    input.dim = dim;
    input.learning_rate = lr;
    input.num_iterations = niter;
    

    OutputBindingInterface output;
    {
        Timer timer;
        output = gradient_descent(input);
    } 

    std::cout << "\t Done: " << output.X << std::endl;
}