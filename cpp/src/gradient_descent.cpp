#include <vector>
#include <Eigen/Dense>

#include "gradient_descent.h"


OutputBindingInterface gradient_descent(InputBindingInterface input){

    // ...

    // write to output
    OutputBindingInterface output;
    output.X.resize(input.N, input.dim);
    output.X = input.X;
    return output;
}