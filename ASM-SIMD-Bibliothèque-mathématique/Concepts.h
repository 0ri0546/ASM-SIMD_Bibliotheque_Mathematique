#pragma once

#include <concepts>

template <typename T>
concept float_num = (std::floating_point<T> || std::same_as<unsigned int, T> || std::same_as<int, T>) && !std::same_as<long double, T>;
