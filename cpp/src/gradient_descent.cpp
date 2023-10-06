#include "gradient_descent.h"

OutputBindingInterface gradient_descent(InputBindingInterface input){
    int value = input.value;

    for (int i=0; i<input.n; i++) {
        value++;
    }
    
    OutputBindingInterface output;
    output.value_out = value;
    output.done = true;

    return output;
}