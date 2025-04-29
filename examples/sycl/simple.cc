#include "drake/examples/sycl/simple.h"

#include <sycl/sycl.hpp>

namespace drake {
namespace examples {
namespace simple {

int sum_ids() {
  for (const auto& platform : sycl::platform::get_platforms()) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>()
              << "\n";
    for (const auto& device : platform.get_devices()) {
      std::cout << "  Device: " << device.get_info<sycl::info::device::name>()
                << " | Type: "
                << (device.is_gpu()   ? "GPU"
                    : device.is_cpu() ? "CPU"
                                      : "Other")
                << " | Vendor: "
                << device.get_info<sycl::info::device::vendor>() << "\n";
    }
  }

  // Creating buffer of 4 ints to be used inside the kernel code
  sycl::buffer<int, 1> Buffer{4};

  // Creating SYCL queue
  sycl::queue Queue{};

  // Size of index space for kernel
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  // Submitting command group(work) to queue
  Queue.submit([&](sycl::handler& cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(NumOfWorkItems, [=](sycl::id<1> WIid) {
      // Fill buffer with indexes
      Accessor[WIid] = static_cast<int>(WIid.get(0));
    });
  });

  // Getting read only access to the buffer on the host.
  // Implicit barrier waiting for queue to complete the work.
  auto HostAccessor = Buffer.get_host_access();

  // Check the results
  bool MismatchFound{false};
  int sum = 0;
  for (size_t I{0}; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != static_cast<int>(I)) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
    sum += HostAccessor[I];
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return sum;
}

}  // namespace simple
}  // namespace examples
}  // namespace drake