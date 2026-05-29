#pragma once

#include <string>
#include "tensor/tensor.h"

namespace minidl {

class MNISTDataset {
public:
    MNISTDataset(DeviceType device = DeviceType::CPU);
    bool load(const std::string& images_path, const std::string& labels_path);
    std::pair<Tensor, Tensor> next_batch(int batch_size);
    void reset();
    int size() const;

private:
    bool load_images(const std::string& path, std::vector<float>& output, int& count, int& rows, int& cols);
    bool load_labels(const std::string& path, std::vector<float>& output, int& count);
    static int read_big_endian_int(std::ifstream& stream);

    Tensor images_;
    Tensor labels_;
    int count_ = 0;
    int position_ = 0;
    DeviceType device_;
};

} // namespace minidl
