#include <vector>

namespace neural_network
{
    template <typename T>
    class Runner
    {
        virtual bool runInferenceSingleIo(const std::vector<T> &input, std::vector<T> &output) = 0;
    };
}