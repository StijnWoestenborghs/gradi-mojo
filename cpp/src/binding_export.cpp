#include <vector>
#include <Eigen/Dense>

#include "binding_export.h"
#include "simpleson.h"
#include "gradient_descent.h"


int run_binding_external(char* c, int length, char** c_return, int* length_return) {
    // Deserialize input_string to json
    std::string input_string = std::string(c, length);
    json::jobject input_json = json::jobject::parse(input_string);

    // Copy input json into Binding struct
    InputBindingInterface input;
    input.N = (int)input_json["N"];
    input.dim = (int)input_json["dim"];
    input.learning_rate = (double)input_json["learning_rate"];
    input.num_iterations = (int)input_json["num_iterations"];

    // Copy (1D) input vector X into matrix of the right shape
    std::vector<float> X_json = (std::vector<float>)input_json["X"];
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
    X.resize(input.N, input.dim); 
    for (int i = 0; i < input.N; i++) {
        for (int j = 0; j < input.dim; j++) {
            X(i, j) = X_json[i * input.dim + j];
        }
    }
    input.X = X;

    // Copy (1D) input vector D into matrix of the right shape
    std::vector<float> D_json = (std::vector<float>)input_json["D"];
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> D;
    D.resize(input.N, input.N);
    for (int i = 0; i < input.N; i++) {
        for (int j = 0; j < input.N; j++) {
            D(i, j) = D_json[i * input.N + j];
        }
    }
    input.D = D;


    // Call the Gradient Descent C++ Function
    OutputBindingInterface output;
    try {
        output = gradient_descent(input);
    
    } catch(...) {
        printf("cppFunction crashed\n");
        return -1;
    }


    // Create output json
    json::jobject output_json;
   
    // copy matrix into (1D) std::vector
    std::vector<double> vector_X(output.X.data(), output.X.data() + output.X.size());
    output_json["X"] = vector_X;

    // Serialize output_json to string
    std::string output_string = (std::string)output_json;

    // Copy output to the output variables
    *c_return = new char[output_string.length()];
    std::copy(output_string.c_str(), output_string.c_str() + output_string.length(), *c_return);
    *length_return = output_string.length();

    return 0;
}

void delete_c_return(char* c_return) {
    delete c_return;
}
