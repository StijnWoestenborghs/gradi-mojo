// Input type definitions
struct InputBindingInterface {
    int N;
    int dim;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> D;
    float learning_rate;
    int num_iterations;
};

// Output type definitions
struct OutputBindingInterface {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
};

// Bounded function
OutputBindingInterface gradient_descent(InputBindingInterface input);
