#include "minidl/framework.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

using namespace minidl;

static int argmax_row(const float* row, int width) {
    int best = 0;
    float max_value = row[0];
    for (int j = 1; j < width; ++j) {
        if (row[j] > max_value) {
            max_value = row[j];
            best = j;
        }
    }
    return best;
}

int main(int argc, char* argv[]) {
    std::string dataset_dir = ".";
    if (argc > 1) {
        dataset_dir = argv[1];
    }
    std::string train_images = dataset_dir + "/train-images-idx3-ubyte";
    std::string train_labels = dataset_dir + "/train-labels-idx1-ubyte";
    std::string test_images = dataset_dir + "/t10k-images-idx3-ubyte";
    std::string test_labels = dataset_dir + "/t10k-labels-idx1-ubyte";

    DeviceType device = DeviceType::CPU;
    int cuda_count = 0;
    if (cudaGetDeviceCount(&cuda_count) == cudaSuccess && cuda_count > 0) {
        device = DeviceType::CUDA;
        std::cout << "CUDA device detected. Running on GPU." << std::endl;
    } else {
        std::cout << "CUDA device unavailable. Running on CPU." << std::endl;
    }

    MNISTDataset train_data(device);
    if (!train_data.load(train_images, train_labels)) {
        std::cerr << "Failed to load training dataset. Ensure MNIST files are available." << std::endl;
        return 1;
    }

    MNISTDataset test_data(device);
    if (!test_data.load(test_images, test_labels)) {
        std::cerr << "Failed to load validation dataset. Ensure MNIST files are available." << std::endl;
        return 1;
    }

    Sequential model;
    model.add(std::make_shared<Linear>(784, 128, device));
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Linear>(128, 10, device));

    std::vector<Tensor*> params = model.parameters();
    Adam optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f);

    const int epochs = 3;
    const int batch_size = 64;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_data.reset();
        int steps = train_data.size() / batch_size;
        float running_loss = 0.0f;

        auto epoch_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) {
            auto [images, labels] = train_data.next_batch(batch_size);
            Tensor logits = model.forward(images);
            Tensor loss = cross_entropy_loss(logits, labels);
            Tensor loss_value = loss.mean(-1);
            running_loss += loss_value.to(DeviceType::CPU).at({});

            model.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto epoch_end = std::chrono::steady_clock::now();
        float avg_loss = running_loss / steps;
        std::cout << "Epoch " << (epoch + 1) << ": loss=" << avg_loss << " time="
                  << std::chrono::duration<double>(epoch_end - epoch_start).count() << "s" << std::endl;
    }

    std::cout << "Validating trained model..." << std::endl;
    int correct = 0;
    int total = 0;
    test_data.reset();
    int test_steps = test_data.size() / batch_size;
    for (int step = 0; step < test_steps; ++step) {
        auto [images, labels] = test_data.next_batch(batch_size);
        Tensor logits = model.forward(images);
        Tensor predictions = logits.softmax(1).to(DeviceType::CPU);
        Tensor batch_labels = labels.to(DeviceType::CPU);

        int width = predictions.shape()[1];
        for (int i = 0; i < predictions.shape()[0]; ++i) {
            int pred = argmax_row(predictions.data() + i * width, width);
            int target = argmax_row(batch_labels.data() + i * width, width);
            correct += (pred == target);
            total += 1;
        }
    }

    std::cout << "Validation accuracy: " << (100.0f * correct / total) << "%" << std::endl;
    return 0;
}
