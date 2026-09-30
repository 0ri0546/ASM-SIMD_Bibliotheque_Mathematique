#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <print>
#include <string>
#include <type_traits>
#include <vector>

#include <intrin.h>

// see https://learn.microsoft.com/en-us/cpp/preprocessor/optimize
// see https://github.com/facebook/folly/blob/v2023.01.30.00/folly/lang/Hint-inl.h#L54-L58
#pragma optimize("", off)
inline void DoNotOptimizeAwaySink(const void*) {}
#pragma optimize("", on)

// see https://github.com/google/benchmark/blob/v1.7.1/include/benchmark/benchmark.h#L514
template <typename T>
__forceinline inline void DoNotOptimizeAway(const T& value) {
	DoNotOptimizeAwaySink(&value);
	_ReadWriteBarrier();
}

inline std::size_t g_benchmark_index = 1;

template <typename Fn>
void Benchmark(const std::string& name, Fn&& fn, std::size_t iterations = 1'000'000, std::size_t samples = 10) {
	using namespace std::chrono;

	const std::size_t per_sample = std::max<std::size_t>(1, iterations / samples);

	std::vector<double> ns_per_op;
	ns_per_op.reserve(samples);
	double total_ns = 0.0;
	std::uint64_t total_cycles = 0;

	for (std::size_t s = 0; s < samples; ++s) {
		const auto c0 = __rdtsc();
		const auto t0 = steady_clock::now();
		for (std::size_t i = 0; i < per_sample; ++i) {
			fn();
			_ReadWriteBarrier();
		}
		const auto t1 = steady_clock::now();
		const auto c1 = __rdtsc();

		const double ns = static_cast<double>(duration_cast<nanoseconds>(t1 - t0).count());
		ns_per_op.push_back(ns / per_sample);
		total_ns += ns;
		total_cycles += c1 - c0;
	}

	const double n = static_cast<double>(ns_per_op.size());
	const double mean = std::accumulate(ns_per_op.begin(), ns_per_op.end(), 0.0) / n;
	double var = 0.0;
	for (double v : ns_per_op) var += (v - mean) * (v - mean);
	var /= (n > 1 ? n - 1 : 1);

	const double err_pct = mean > 0 ? 100.0 * std::sqrt(var) / mean : 0.0;

	std::println("{:<4} | {:<40} | {:>12.4e} | {:>12.4f} | {:>12.4f} | {:>12} | {:>12.4f}",
		g_benchmark_index++,
		name,
		1.0e9 / mean,                          // op/s
		mean,                                  // ns/op
		err_pct,                               // err %
		total_cycles / (per_sample * samples), // cycles/op (TSC ticks)
		total_ns * 1.0e-6);                    // total ms
	std::println("{:-<122}", "");
}
