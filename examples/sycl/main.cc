#include "drake/examples/sycl/simple.h"
#include <fmt/format.h>

int main() {
  const int result = drake::examples::simple::sum_ids();
  fmt::print("Result from calling sycl function: {}\n", result);
  return 0;
}