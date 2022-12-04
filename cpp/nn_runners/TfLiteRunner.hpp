#include "RunnerInterface.hpp"

#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/model_builder.h>
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/c/common.h"

#include <fstream>
#include <iostream>

namespace neural_network
{
    namespace tflite
    {
        using TfLiteRunnerType = float;

        class TfLiteRunner : public Runner<TfLiteRunnerType>
        {
        private:
            std::unique_ptr<::tflite::FlatBufferModel> m_flatbufferModelPtr;
            std::unique_ptr<::tflite::Interpreter> m_interpreterPtr;

        public:
            TfLiteRunner(const std::string &model_path) : m_flatbufferModelPtr{::tflite::FlatBufferModel::BuildFromFile(model_path.c_str())},
                                                          m_interpreterPtr{}
            {
                if (!m_flatbufferModelPtr)
                {
                    throw std::runtime_error("TfLite model construction failed!. Please ensure the model is valid (and the path exists!");
                }
                ::tflite::ops::builtin::BuiltinOpResolver resolver;
                if (::tflite::InterpreterBuilder(*m_flatbufferModelPtr, resolver)(&m_interpreterPtr) != kTfLiteOk)
                {
                    throw std::runtime_error("TfLite interpreter building failed!");
                }

                if (m_interpreterPtr->AllocateTensors() != kTfLiteOk)
                {
                    throw std::runtime_error("Allocating memory for the TfLite interpreter failed!");
                }
                const auto success = m_interpreterPtr->Invoke() == kTfLiteOk;

                if (!success)
                {
                    std::cerr << "Invoking failed" << std::endl;
                }
            }

            ~TfLiteRunner() = default;

            // Implemetation of the Runner interface
            bool runInferenceSingleIo(const std::vector<TfLiteRunnerType> &input, std::vector<TfLiteRunnerType> &output) override
            {
                const auto firstInputOutputIndex{0};

                auto firstInputTensorPtr = m_interpreterPtr->input_tensor(firstInputOutputIndex);
                auto firstInputTensorDataPtr = m_interpreterPtr->typed_input_tensor<TfLiteRunnerType>(firstInputOutputIndex);

                auto success = firstInputTensorPtr->bytes >= input.size() * sizeof(TfLiteRunnerType);

                if (success)
                {
                    std::memcpy(firstInputTensorDataPtr, input.data(), input.size());

                    success = m_interpreterPtr->Invoke();
                }
                else
                {
                    std::cerr << "The interpreter's input tensor is smaller than the provided input! No automatic resizing attempted." << std::endl;
                }

                if (success)
                {
                    auto firstOutputTensorPtr = m_interpreterPtr->output_tensor(firstInputOutputIndex);
                    auto firstOutputTensorDataPtr = m_interpreterPtr->typed_output_tensor<TfLiteRunnerType>(firstInputOutputIndex);

                    if (output.size() * sizeof(TfLiteRunnerType) < firstInputTensorPtr->bytes)
                    {
                        output.reserve(firstOutputTensorPtr->bytes);
                    }

                    std::memcpy(output.data(), firstOutputTensorDataPtr, firstOutputTensorPtr->bytes);
                }
                else
                {
                    std::cerr << "Invoking the interpreter failed!!" << std::endl;
                }
                return success;
            }
        };

    }
}