#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <omp.h>

#include "gradient_descent.h"


void compute_gradient(
        Eigen::MatrixXf & grad,
        const Eigen::MatrixXf & X,
        const Eigen::MatrixXf & D
    ) {
    float squared_distance;
    
    int num_threads = omp_get_max_threads();
    // std::cout << num_threads << std::endl;     // 20 for my machine
    omp_set_num_threads(num_threads);
    #pragma omp parallel for private(squared_distance)
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.rows(); ++j) {
            squared_distance = 0;
            for (int d = 0; d < X.cols(); ++d) {
                squared_distance += (X(i, d) - X(j, d)) * (X(i, d) - X(j, d));
            }

            for (int d = 0; d < X.cols(); ++d) {
                grad(i, d) += 4 * (squared_distance - D(i, j) * D(i, j)) * (X(i, d) - X(j, d));
            }
        }
    }
}


OutputBindingInterface gradient_descent(InputBindingInterface input){
    Eigen::MatrixXf grad(input.N, input.dim);
    Eigen::MatrixXf X = input.X;
    
    for (int iter = 0; iter < input.num_iterations; ++iter) {
        grad.setZero();
        compute_gradient(grad, X, input.D);

        for (int r = 0; r < X.rows(); ++r) {
            for (int c = 0; c < X.cols(); ++c) {
                X(r, c) -= input.learning_rate * grad(r, c);
            }
        }
    }

    // write to output
    OutputBindingInterface output;
    output.X.resize(input.N, input.dim);
    output.X = X;
    return output;
}