#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::uint64_t modulus;
int tuple_size;
std::vector<std::vector<std::uint64_t>> term;
std::vector<std::uint32_t> histogram;

void enumerate(int rank, int upper_exclusive, std::uint64_t residue) {
    if (rank == tuple_size) {
        if (histogram[residue] == std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << "histogram bin overflow\n";
            std::exit(2);
        }
        ++histogram[residue];
        return;
    }

    const int remaining_after_this = tuple_size - rank - 1;
    for (int exponent = upper_exclusive - 1;
         exponent >= remaining_after_this;
         --exponent) {
        enumerate(rank + 1, exponent,
                  (residue + term[rank][exponent]) % modulus);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: histogram L M OUTPUT.u32\n";
        return 2;
    }

    const int l = std::stoi(argv[1]);
    tuple_size = std::stoi(argv[2]);
    const std::string output_path = argv[3];
    if (l < 1 || tuple_size < 1) {
        std::cerr << "L and M must be positive\n";
        return 2;
    }

    modulus = 1;
    for (int i = 0; i < l; ++i) {
        if (modulus > std::numeric_limits<std::uint64_t>::max() / 3) {
            std::cerr << "3^L does not fit in uint64_t\n";
            return 2;
        }
        modulus *= 3;
    }
    if (modulus > std::numeric_limits<std::size_t>::max()) {
        std::cerr << "histogram is too large for this platform\n";
        return 2;
    }

    std::vector<std::uint64_t> powers_of_two(2 * tuple_size, 1);
    for (int a = 1; a < 2 * tuple_size; ++a) {
        powers_of_two[a] = (2 * powers_of_two[a - 1]) % modulus;
    }
    std::vector<std::uint64_t> powers_of_three(tuple_size, 1);
    for (int i = 1; i < tuple_size; ++i) {
        powers_of_three[i] = (3 * powers_of_three[i - 1]) % modulus;
    }

    term.assign(tuple_size, std::vector<std::uint64_t>(2 * tuple_size));
    for (int i = 0; i < tuple_size; ++i) {
        for (int a = 0; a < 2 * tuple_size; ++a) {
            const __uint128_t product =
                static_cast<__uint128_t>(powers_of_two[a]) * powers_of_three[i];
            term[i][a] = static_cast<std::uint64_t>(product % modulus);
        }
    }

    histogram.assign(static_cast<std::size_t>(modulus), 0);
    enumerate(0, 2 * tuple_size, 0);

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::cerr << "could not open output file\n";
        return 2;
    }
    output.write(reinterpret_cast<const char*>(histogram.data()),
                 static_cast<std::streamsize>(histogram.size() * sizeof(histogram[0])));
    if (!output) {
        std::cerr << "could not write complete histogram\n";
        return 2;
    }
}
