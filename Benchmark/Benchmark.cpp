#pragma once

#include "Benchmark.h"

#include <filesystem>
#include "Headers/Batch.h"
#include "BatchSIMD.h"

void BenchmarkNoSimd()
{
    std::println();
    std::println();
    std::println();
    std::println("==============================================================");
    std::println("|          BENCHMARK WITHOUT SIMD OPTIMIZATIONS              |");
    std::println("==============================================================");
    std::println();
    PrintBenchmarkHeader();
    BenchmarkVecNoSIMD();
    BenchmarkMatNoSIMD();
    BenchmarkQuaternionNoSIMD();
    BenchmarkBatchNoSIMD();
    std::println("{:-<180}", "");
}

void BenchmarkSIMD()
{
    std::println();
    std::println();
    std::println();
    std::println("==============================================================");
    std::println("|          BENCHMARK WITH SIMD OPTIMIZATIONS                 |");
    std::println("==============================================================");
    std::println();
    PrintBenchmarkHeader();
    BenchmarkVecSIMD();
    BenchmarkMatSIMD();
    BenchmarkQuaternionSIMD();
    BenchmarkBatchSIMD();
    std::println("{:-<180}", "");
}

int main()
{
    if (!SetTerminalColors()) return -1;

    if (std::filesystem::exists("Benchmark.csv"))
    {
        std::filesystem::remove("Benchmark.csv");
    }

    std::println();
    std::println("==============================================================");
    std::println("|                 CONFIGURATION DU BENCHMARK                 |");
    std::println("==============================================================");
    std::println();

    std::size_t batchSize = ChooseBatchSize();
    int benchmarkType = ChooseBenchmarkType();

    constexpr std::uint32_t seed = 0x12345678;

    BeginBenchmark(batchSize, seed);

    if (benchmarkType == 1 || benchmarkType == 3)
    {
        BenchmarkNoSimd();
    }

    if (benchmarkType == 2 || benchmarkType == 3)
    {
        BenchmarkSIMD();
    }

    if (benchmarkType == 3)
    {
        CalculateSpeedups();
        PrintSpeedupTable();
    }

    std::println();
    std::println("Resultats sauvegardes dans Benchmark.csv");

    return 0;
}