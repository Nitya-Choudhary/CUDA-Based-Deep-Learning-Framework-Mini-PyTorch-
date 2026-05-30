#include "datasets/mnist.h"
#include "cuda/cuda_utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

namespace minidl {

MNISTDataset::MNISTDataset(DeviceType device)
    : device_(device) {}

int MNISTDataset::read_big_endian_int(std::ifstream& stream) {
    int value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    unsigned char* bytes = reinterpret_cast<unsigned char*>(&value);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

bool MNISTDataset::load_images(const std::string& path, std::vector<float>& output, int& count, int& rows, int& cols) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    int magic = read_big_endian_int(input);
    if (magic != 2051) return false;
    count = read_big_endian_int(input);
    rows = read_big_endian_int(input);
    cols = read_big_endian_int(input);
    output.resize(count * rows * cols);
    for (int i = 0; i < count * rows * cols; ++i) {
        unsigned char value = 0;
        input.read(reinterpret_cast<char*>(&value), 1);
        output[i] = static_cast<float>(value) / 255.0f;
    }
    return true;
}

bool MNISTDataset::load_labels(const std::string& path, std::vector<float>& output, int& count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    int magic = read_big_endian_int(input);
    if (magic != 2049) return false;
    count = read_big_endian_int(input);
    output.assign(count * 10, 0.0f);
    for (int i = 0; i < count; ++i) {
        unsigned char label;
        input.read(reinterpret_cast<char*>(&label), 1);
        output[i * 10 + static_cast<int>(label)] = 1.0f;
    }
    return true;
}

bool MNISTDataset::load(const std::string& images_path, const std::string& labels_path) {
    std::vector<float> image_data;
    std::vector<float> label_data;
    int rows = 0, cols = 0;
    int image_count = 0, label_count = 0;

    if (!load_images(images_path, image_data, image_count, rows, cols)) {
        std::cerr << "Failed to open MNIST images file: " << images_path << std::endl;
        return false;
    }
    if (!load_labels(labels_path, label_data, label_count)) {
        std::cerr << "Failed to open MNIST labels file: " << labels_path << std::endl;
        return false;
    }
    if (image_count != label_count) {
        std::cerr << "MNIST image and label counts do not match." << std::endl;
        return false;
    }

    count_ = image_count;
    position_ = 0;
    images_ = Tensor::fromVector(image_data, {count_, rows * cols}, DeviceType::CPU, false);
    labels_ = Tensor::fromVector(label_data, {count_, 10}, DeviceType::CPU, false);
    return true;
}

std::pair<Tensor, Tensor> MNISTDataset::next_batch(int batch_size) {
    if (position_ >= count_) {
        reset();
    }
    int size = std::min(batch_size, count_ - position_);
    std::vector<int> image_shape = {size, images_.shape()[1]};
    std::vector<int> label_shape = {size, labels_.shape()[1]};

    Tensor batch_images(image_shape, device_, false);
    Tensor batch_labels(label_shape, device_, false);

    if (device_ == DeviceType::CPU) {
        std::memcpy(batch_images.data(), images_.data() + position_ * images_.shape()[1], size * images_.shape()[1] * sizeof(float));
        std::memcpy(batch_labels.data(), labels_.data() + position_ * labels_.shape()[1], size * labels_.shape()[1] * sizeof(float));
    } else {
        std::vector<float> local_images(size * images_.shape()[1]);
        std::vector<float> local_labels(size * labels_.shape()[1]);
        std::memcpy(local_images.data(), images_.data() + position_ * images_.shape()[1], size * images_.shape()[1] * sizeof(float));
        std::memcpy(local_labels.data(), labels_.data() + position_ * labels_.shape()[1], size * labels_.shape()[1] * sizeof(float));
        cuda_ops::check_cuda(cudaMemcpy(batch_images.data(), local_images.data(), size * images_.shape()[1] * sizeof(float), cudaMemcpyHostToDevice));
        cuda_ops::check_cuda(cudaMemcpy(batch_labels.data(), local_labels.data(), size * labels_.shape()[1] * sizeof(float), cudaMemcpyHostToDevice));
    }

    position_ += size;
    return {batch_images, batch_labels};
}

void MNISTDataset::reset() {
    position_ = 0;
}

int MNISTDataset::size() const {
    return count_;
}

} // namespace minidl
