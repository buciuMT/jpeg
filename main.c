#include <stdio.h>
#define __BSTD_IMPLEMENT_ALL__
#include "bstd_all.h"

int main(int argc, const char *argv[]) {
  string input_file, output_file;
  bstd_args arg_parser = {
      .args =
          {
              {.type = BS_ARG_STRING, .shortform = 'i', .data = &input_file},
              {.type = BS_ARG_STRING, .shortform = 'o', .data = &output_file},
          },
      .count = 2};
  bstd_args_parse(&arg_parser, argc - 1, argv + 1, NULL);
  printf("%s", input_file.str);
  return 0;
}
