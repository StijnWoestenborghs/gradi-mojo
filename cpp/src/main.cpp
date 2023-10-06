#include <iostream>
#include <Eigen/Dense>
#include <cmath>

#include "timer.h"
#include "gradient_descent.h"


const int N = 10;
const int dim = 2;
const float lr = 0.00001;
const int niter = 1000;


Eigen::Matrix<float, N, dim> generate_radial_points() {
    Eigen::Matrix<float, N, dim> points;
    const float r = 3.0;
    float angle;
    

    if (dim == 2) {
        for (int i = 0; i < N; i++) {
            angle = 2 * M_PI * i / N;
            points(i, 0) = r * std::cos(angle);
            points(i, 1) = r * std::sin(angle);
        }
    } else if (dim == 3) {
        for (int i = 0; i < N; i++) {
            float phi = std::acos(1 - 2 * static_cast<float>(i) / N);
            float theta = std::sqrt(N * M_PI) * phi;
            points(i, 0) = r * std::sin(phi) * std::cos(theta);
            points(i, 1) = r * std::sin(phi) * std::sin(theta);
            points(i, 2) = r * std::cos(phi);
        }
    } else {
        std::cerr << "Only supports 2D and 3D" << std::endl;
        exit(1);
    }
    
    return points;
}


Eigen::Matrix<float, N, N> generate_distance_matrix(const Eigen::Matrix<float, N, dim>& points) {
    Eigen::Matrix<float, N, N> distance_matrix = Eigen::Matrix<float, N, N>::Zero();
    
    for (int i = 0; i < N; i++) {
        for (int j = i+1; j < N; j++) {
            float distance = (points.row(i) - points.row(j)).norm();
            distance_matrix(i, j) = distance;
            distance_matrix(j, i) = distance;
        }
    }

    return distance_matrix;
}



int main(){
    std::cout << "[EXECUTABLE] Gradient Descent as C++ executable " << std::endl;

    // generate optimization target
    Eigen::Matrix<float, N, dim> points = generate_radial_points();
    Eigen::Matrix<float, N, N> D = generate_distance_matrix(points);
    
    // Initial starting point
    Eigen::Matrix<float, N, dim> X = Eigen::Matrix<float, N, dim>::Random();   // Between -1 and 1

    InputBindingInterface input;
    input.N = N;
    input.dim = dim;
    input.learning_rate = lr;
    input.num_iterations = niter;
    input.X = X;
    input.D = D;


    OutputBindingInterface output;
    {
        Timer timer;
        output = gradient_descent(input);
    } 


    std::cout << "\t Done: \n" << output.X << std::endl;
}