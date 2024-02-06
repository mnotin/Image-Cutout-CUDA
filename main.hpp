#ifndef MAIN_HPP
#define MAIN_HPP

#define MATRIX_SIZE_PER_BLOCK 16

enum ProcessingUnit { Host, Device };

void print_help(char *app_name);
void print_bad_usage(char *app_name);

#endif // MAIN_HPP
