#pragma once

#include <chrono>
#include <numeric>
#include <print>
#include <string>

#include <intrin.h> // for _ReadWriteBarrier

namespace {
	inline std::size_t benchmarks_count = 1;
	inline const std::size_t bench_calls = 5;
}

template <typename T>
inline void DoNotOptimize(const T& value) {
	// Address escapes into a volatile: the compiler must materialize `value` in memory
	static const volatile void* volatile sink;
	sink = &value;
	_ReadWriteBarrier(); // compiler-level barrier (deprecated but still works on MSVC)
}

template <typename Fn>
void Benchmark(const std::string& name, Fn&& fn, std::size_t iterations = 1'000'000) {
	using namespace std::chrono;
	auto start = high_resolution_clock::now();
	for (std::size_t i = 0; i < iterations; ++i) {
		fn();
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<nanoseconds>(end - start).count();
	double avg_time = static_cast<double>(duration) / iterations;
	std::println("{:<4} {:<40} {:>20.3f} {:>20.3f}", benchmarks_count, name, duration * 1.0e-6, avg_time);
	std::println();
	benchmarks_count++;
}
