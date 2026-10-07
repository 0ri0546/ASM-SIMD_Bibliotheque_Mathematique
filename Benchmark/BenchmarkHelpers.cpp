#include "Benchmark.h"

std::size_t ChooseBatchSize()
{
    std::println("Choisir la taille du lot :");
    std::println();
    std::println("1. 1 000");
    std::println("2. 100 000");
    std::println("3. 1 000 000");
    std::println("4. Personnalise");
    std::println();

    int choice = 0;

    std::cout << "Choix : ";
    std::cin >> choice;

    if (choice == 1)
        return 1'000;
    if (choice == 2)
        return 100'000;
    if (choice == 3)
        return 1'000'000;
    if (choice == 4) {
        std::size_t size = 0;

        std::cout << "Taille du lot : ";
        std::cin >> size;

        if (size > 0)
            return size;

        std::println("error, on utilise 1000");
        return 1'000;
    }

    std::println("error, on utilise 1000");
    return 1'000;
}

int ChooseBenchmarkType()
{
    std::println();
    std::println("Choisir les benchmarks :");
    std::println();
    std::println("1. Reference C++");
    std::println("2. SIMD");
    std::println("3. Les deux");
    std::println();

    int choice = 0;

    std::cout << "Choix : ";
    std::cin >> choice;

    if (choice >= 1 && choice <= 3)
        return choice;

    std::println("Choix invalide. Les deux seront fait.");
    return 3;
}

bool SetTerminalColors() {
    const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(h, &mode))
        return false;
    if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        return false;
    return true;
}

std::string ANSI_RGB(int r, int g, int b) {
    return std::format("\x1B[38;2;{};{};{}m", r, g, b);
}
