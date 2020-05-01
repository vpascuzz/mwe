//
// Test.cc
//

#include "Test.h"

bool Test::AllocMem(cl::sycl::device* dev) {
  eles_ = (unsigned int*)malloc(kNumElements * sizeof(int));
  if (!eles_) {
    std::cout << "Cannot allocate host-side memory!\n";
    return false;
  }
  eles_device_ =
      (unsigned int*)malloc_device(kNumElements * sizeof(int), *dev, ctx_);
  if (!eles_device_) {
    std::cout << "Cannot allocate device-side memory!\n";
    return false;
  }
  return true;
}

bool Test::LoadToDevice() {
// Device, queue and context setup
#ifdef CPU_DEVICE
  cl::sycl::cpu_selector dev_selector;
#elif GPU_DEVICE
  cl::sycl::gpu_selector dev_selector;
#elif CUDA_DEVICE
  CUDASelector dev_selector;
#else
  cl::sycl::default_selector dev_selector;
#endif
  cl::sycl::device dev(dev_selector);
  cl::sycl::queue queue(dev);
  ctx_ = queue.get_context();
  // Name of the device to run on
  std::string dev_name =
      queue.get_device().get_info<cl::sycl::info::device::name>();
  std::cout << "Using device \"" << dev_name << "\"" << std::endl;

  // Ensure device can handle USM device allocations.
  if (!queue.get_device()
           .get_info<cl::sycl::info::device::usm_device_allocations>()) {
    std::cout << "ERROR :: device \"" << dev_name
              << "\" does not support usm_device_allocations!" << std::endl;
    return false;
  }

  // Allocate memory
  if (!AllocMem(&dev)) {
    std::cout << "Could not allocate host- or device-side memory!" << std::endl;
    return false;
  }

  // Fill host-side memory with dummy Elements
  for (unsigned int iel = 0; iel < kNumElements; ++iel) {
    eles_[iel] = iel;
  }

  // Copy host memory to device
  auto ev_cpy_cells =
      queue.memcpy(eles_device_, &eles_[0], kNumElements * sizeof(int));
  ev_cpy_cells.wait_and_throw();

#ifndef CUDA_DEVICE
  // Read device memory, unless a CUDA device (which currently don't support
  // experimental::printf())
  std::cout << "Test device cells..." << std::endl;
  auto ev_cellinfo = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<class Dummy>(
        cl::sycl::range<1>(kNumElements),
        [=, dev_cells_local = this->eles_device_](cl::sycl::id<1> idx) {
          unsigned int id = (int)idx[0];
          int val = dev_cells_local[id];
          cl::sycl::intel::experimental::printf(kPrintf, id, val);
        });
  });
  ev_cellinfo.wait_and_throw();
#endif  // !CUDA_DEVICE

  return true;
}
