#include <iostream>

#include "timer.h"
#include "gradient_descent.h"



int main(){
    std::cout << "[EXECUTABLE] Gradient Descent as C++ executable " << std::endl;

    InputBindingInterface input;
    input.value = 0;
    input.n = 1000000000;
    
    OutputBindingInterface output;
    {
        Timer timer;
        output = gradient_descent(input);
    } 

    std::cout << "\t Done: " << output.done << std::endl;
    std::cout << "\t Value out: " << output.value_out << std::endl;
}