#include <nn_runners/RunnerInterface.hpp>
#include <nn_runners/TfLiteRunner.hpp>

#include <string>
#include <cstdlib>
#include <map>

// NN paths.
constexpr auto PRECLASSIFIER_PATH = "../models/hulks_2022/preclassifier.hdf5";
constexpr auto POSITIONER_PATH = "../models/hulks_2022/positioner.hdf5";

constexpr auto CLASSIFIER_PATH = "../models/hulks_2022/classifier.hdf5";
constexpr auto CLASSIFIER_PATH_ONNX = "../models/hulks_2022/classifier.onnx";
constexpr auto CLASSIFIER_PATH_TFLITE = "../models/hulks_2022/classifier.tflite";

constexpr auto BALL_SAMPLE_PATH = "../data/ball_sample.png";


int main(int argc, char **argv)
{
const auto image
    return EXIT_SUCCESS;
}