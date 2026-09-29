#pragma once

#include <chrono>
#include <print>
#include <string>

namespace {
	inline std::size_t benchmarks_count = 1;
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
