#include "device_scene.h"

#include <cstring>
#include <utility>

DeviceBuffer::~DeviceBuffer() {
  reset();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept {
  data_ = other.data_;
  size_ = other.size_;
  other.data_ = nullptr;
  other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void DeviceBuffer::allocate(std::size_t bytes) {
  reset();
  if (bytes == 0) {
    return;
  }
  CUDA_CHECK(cudaMalloc(&data_, bytes));
  size_ = bytes;
}

void DeviceBuffer::upload(const void* data, std::size_t bytes) {
  if (bytes == 0) {
    reset();
    return;
  }
  if (size_ < bytes || data_ == nullptr) {
    allocate(bytes);
  }
  CUDA_CHECK(cudaMemcpy(data_, data, bytes, cudaMemcpyHostToDevice));
}

void DeviceBuffer::reset() {
  if (data_ != nullptr) {
    CUDA_CHECK(cudaFree(data_));
    data_ = nullptr;
    size_ = 0;
  }
}

void DeviceScene::adopt(DeviceBuffer primitives,
                        DeviceBuffer materials,
                        Metadata metadata) {
  primitives_ = std::move(primitives);
  materials_ = std::move(materials);
  metadata_ = metadata;
}

DeviceSceneView DeviceScene::view() const {
  DeviceSceneView v;
  v.primitives = reinterpret_cast<const DevicePrimitive*>(primitives_.data());
  v.materials = reinterpret_cast<const DeviceMaterial*>(materials_.data());
  v.primitive_count = metadata_.primitive_count;
  v.material_count = metadata_.material_count;
  v.light_primitive = metadata_.light_primitive;
  return v;
}
