// Tiny Llama Gen 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    time_t start_time;
    struct tm *start_time_struct;

    start_time = time(NULL);
    start_time_struct = localtime(&start_time);

    printf("Loading...\n");

    getchar(); // wait for keypress

    printf("Loading complete.\n");

    printf("Press any key to continue...\n");

    getchar();

    return 0;
}
