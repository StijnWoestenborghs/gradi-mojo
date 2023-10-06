// Input type definitions
struct InputBindingInterface {
    int value;
    int n;
};

// Output type definitions
struct OutputBindingInterface {
    int value_out;
    bool done;
};

// Bounded function
OutputBindingInterface gradient_descent(InputBindingInterface input);
