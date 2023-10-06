extern "C" {

    extern int run_binding_external(char* c, int length, char** c_return, int* c_length);

    extern void delete_c_return(char* c_return);

}