#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/multiply_cl.h"

#include <vector>
#include <iostream>
#include <random>

/**
 * Считает вероятность генерации единицы, чтобы после умножения матриц, где промежуточная размерность равна size,
 * получилась вероятность prob получить единицу.
*/
double estimateProbability(double prob, uint32_t size) {
    return sqrt(1.0 - pow((1.0 - prob), 1.0 / size));
}

/**
 * Генерируем случайную булеву матрицу размером dim1 x dim2. Ставим единицу с вероятностью prob
 */
std::vector<std::vector<unsigned>> generateRandomVector(size_t dim1, size_t dim2, double prob) {
    std::random_device randomDevice;
    std::mt19937_64 generator(randomDevice());
    std::vector<std::vector<unsigned>> vector(dim1, std::vector<unsigned>(dim2));
    std::bernoulli_distribution coin(prob);
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            vector[i][j] = coin(generator);
        }
    }
    return vector;
}

/**
 * Используем алгоритм Флойда, чтобы получить наивное транзитивное замыкание.
 */
std::vector<std::vector<unsigned>> closureFloid(const std::vector<std::vector<unsigned>> &vector, size_t size) {
    std::vector<std::vector<unsigned>> result(vector);
    for (size_t i = 0; i < size; i++) {
        result[i][i] = true;
    }
    for (size_t k = 0; k < size; k++) {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                result[i][j] = result[i][j] || (result[i][k] && result[k][j]);
            }
        }
    }
    return result;
}

/**
 * Конвертирует двумерную матрицу в одномерную.
 */
std::vector<unsigned> convert2oneDim(const std::vector<std::vector<unsigned>> &data) {
    std::vector<unsigned> result;
    for (const auto &vec : data) {
        for (unsigned item : vec) {
            result.push_back(item);
        }
    }
    return result;
}

/**
 * Конвертирует одномерную матрицу в двумерную.
 */
std::vector<std::vector<unsigned>> convert2twoDim(const std::vector<unsigned> &data, size_t dim1, size_t dim2) {
    std::vector<std::vector<unsigned>> result(dim1);
    for (size_t i = 0; i < dim1; i++) {
        for (size_t j = 0; j < dim2; j++) {
            result[i].push_back(data[dim2 * i + j]);
        }
    }
    return result;
}

/**
 * Перемножает булеву матрицу a[M x K] на матрицу b[K x N] наивным образом.
 */
std::vector<std::vector<unsigned>> matrix_multiplication_obvious(const std::vector<std::vector<unsigned>> &a,
                                                                 const std::vector<std::vector<unsigned>> &b,
                                                                 unsigned M, unsigned N, unsigned K) {
    std::vector<std::vector<unsigned>> result(M, std::vector<unsigned>(N, false));
    for (unsigned i = 0; i < M; i++) {
        for (unsigned j = 0; j < N; j++) {
            for (unsigned k = 0; k < K; k++) {
                result[i][j] = result[i][j] || (a[i][k] && b[k][j]);
            }
        }
    }
    return result;
}

/**
 * Сравнивает матрицы и, если они различны, выводит ошибку в человекочитаемом виде. Возвращает код возврата из main.
 */
int compare_matrixes(const std::vector<std::vector<unsigned>> &a, const std::vector<std::vector<unsigned>> &b) {
    if (a.size() != b.size()) {
        std::cerr << "Not eq size: " << a.size() << ' ' << b.size() << std::endl;
        return 1;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) {
            std::cerr << "Not eq subsize[" << i << "]: " << a[i].size() << ' ' << b[i].size() << std::endl;
            return 1;
        }
    }
    std::stringstream ss1, ss2;
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a[i].size(); j++) {
            ss1 << a[i][j];
            ss2 << b[i][j];
            if (a[i][j] != b[i][j]) {
                std::cerr << "Not eq arrays in last position:\n"
                          << ss1.str() << "... <---\n\n" << ss2.str() << "... <---\n\n";
                return 1;
            }
            ss1 << ' ';
            ss2 << ' ';
        }
        ss1 << '\n';
        ss2 << '\n';
    }
    return 0;
}

/**
 * Печатает матрицу в stdin.
 */
void print_matrix(const std::vector<std::vector<unsigned>> &matrix) {
    for (const auto &v: matrix) {
        for (const auto &item: v) {
            std::cout << item << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    /**настройка контекста, компилирование kernel-а**/
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel multiply_kernel(multiply, multiply_length,
                                             "multiply");
    multiply_kernel.compile();

    /**повторяем тест умножения несколько раз на разных размерах**/
    std::cout << "--- Multiply test set ---" << std::endl;
    for (unsigned i = 7; i <= 11; i++) {
        /**размерности матриц**/
        unsigned M = (1U << i) + 7;
        unsigned N = (1U << i) - 7;
        unsigned K = (1U << i) + 1;

        /**сами сгенерированные матрицы**/
        auto as_2 = generateRandomVector(M, K, estimateProbability(0.5, K));
        auto bs_2 = generateRandomVector(K, N, estimateProbability(0.5, K));
        double time = clock();
        auto cs_2 = matrix_multiplication_obvious(as_2, bs_2, M, N, K);
        std::cout << "CPU work time: " << (clock() - time) / CLOCKS_PER_SEC * 1000 << " millis\n";

        /**они же самые в одномерном виде**/
        auto as = convert2oneDim(as_2);
        auto bs = convert2oneDim(bs_2);
        auto cs = convert2oneDim(cs_2);

        /**буффер, куда считаем ответ из видеокарты**/
        std::vector<unsigned> storage(cs.size());

        time = clock();

        /**записываем все в буфферы, для отправки на gpu**/
        gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu;
        as_gpu.resizeN(as.size());
        bs_gpu.resizeN(bs.size());
        cs_gpu.resizeN(storage.size());

        as_gpu.writeN(as.data(), as.size());
        bs_gpu.writeN(bs.data(), bs.size());

        /**размер тайла на gpu, выравнивание размеров рабочего пространства**/
        const unsigned tile_size = 16;
        const unsigned global_work_size_m = (M + tile_size - 1) / tile_size * tile_size;
        const unsigned global_work_size_n = (N + tile_size - 1) / tile_size * tile_size;
        /**исполнение kernel-а**/
        multiply_kernel.exec(
                gpu::WorkSize(tile_size, tile_size, global_work_size_n, global_work_size_m), as_gpu, bs_gpu,
                cs_gpu, M, K, N);

        /**считываем результат**/
        cs_gpu.readN(storage.data(), storage.size());

        std::cout << "GPU work time: " << (clock() - time) / CLOCKS_PER_SEC * 1000 << " millis\n";

        /**конвертируем результат в 2D**/
        auto result = convert2twoDim(storage, M, N);

        /**проверяем на правильность**/
        int err = compare_matrixes(cs_2, result);
        if (err != 0) {
            return err;
        }
        std::cout << "Correct results in multiply test #" << i << "\n" << std::endl;
    }

    /**повторяем тест транзитивного замыкания несколько раз на разных размерах**/
    std::cout << "--- Transitivity closure test set ---" << std::endl;
    for (unsigned j = 7; j <= 11; j++) {
        /**количество вершин в графе**/
        unsigned N = (1U << j) + 1;
        /**вычисление вероятности поставить единичку. Не знал как адекватно оценить.
         * В итоге повторение 4 раза estimateProbability сработало лучше всего на разных размерах данных
         * (получался хороший балланс единиц и нулей).
         */
        double probability = 0.5;
        for (unsigned i = 0; i < 4; i++) {
            probability = estimateProbability(probability, N);
        }

        /**генерируем матрицу смежности графа**/
        auto graph_matrix = generateRandomVector(N, N, probability);
        /**вычисляем замыкание на cpu**/
        double time = clock();
        auto closed_matrix = closureFloid(graph_matrix, N);
        std::cout << "CPU work time: " << (clock() - time) / CLOCKS_PER_SEC * 1000 << " millis\n";

        time = clock();

        /**матрица, которая после gpu станет матрицой транзитивного замыкания графа**/
        std::vector<std::vector<unsigned>> matrix_2(graph_matrix);
        /**подготовка к отправлению на gpu: необходимо на всех вершинах поставить петли**/
        for (unsigned i = 0; i < N; i++) {
            matrix_2[i][i] = true;
        }
        /**конвертируем в 1D**/
        auto matrix = convert2oneDim(matrix_2);

        /**буффер для результата**/
        std::vector<unsigned> storage(matrix.size());


        /**записываем все в буфферы, для отправки на gpu**/
        gpu::gpu_mem_32u matrix_gpu1, matrix_gpu2;
        matrix_gpu1.resizeN(matrix.size());
        matrix_gpu2.resizeN(matrix.size());

        matrix_gpu1.writeN(matrix.data(), matrix.size());

        /**количество раз, которое нужно перемножить будущую матрицу транзитивного замыкания (очевидно, логарифм)**/
        unsigned mul_number = log2(N) + 1;
        for (unsigned i = 0; i < mul_number; i++) {
            /**то же самое, что и в прошлых тестах**/
            unsigned tile_size = 16;
            unsigned global_work_size = (N + tile_size - 1) / tile_size * tile_size;
            /**перемножаем сначала то, что лежит в matrix_gpu1, результат записываем в matrix_gpu2, затем меняем их ролями
             * без выгрузки из видеопамяти
            **/
            if (i % 2 == 0) {
                /**исполнение kernel-а**/
                multiply_kernel.exec(
                        gpu::WorkSize(tile_size, tile_size, global_work_size, global_work_size),
                        matrix_gpu1, matrix_gpu1, matrix_gpu2, N, N, N);
            } else {
                /**исполнение kernel-а**/
                multiply_kernel.exec(
                        gpu::WorkSize(tile_size, tile_size, global_work_size, global_work_size),
                        matrix_gpu2, matrix_gpu2, matrix_gpu1, N, N, N);
            }
        }
        /**если mul_number четное, то результат в matrix_gpu1, иначе в matrix_gpu2**/
        if (mul_number % 2 == 0) {
            matrix_gpu1.readN(storage.data(), storage.size());
        } else {
            matrix_gpu2.readN(storage.data(), storage.size());
        }

        std::cout << "GPU work time: " << (clock() - time) / CLOCKS_PER_SEC * 1000 << " millis\n";

        /**конвертируем в 2D**/
        auto result = convert2twoDim(storage, N, N);

        /**проверяем на правильность**/
        int err = compare_matrixes(closed_matrix, result);
        if (err != 0) {
            return err;
        }
        std::cout << "Correct results in transitivity closure test #" << j << "\n" << std::endl;
    }

    return 0;
}
