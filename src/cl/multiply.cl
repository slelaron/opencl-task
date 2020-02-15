#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define TILE_SIZE 16

__kernel void
multiply(__global unsigned *a, __global unsigned *b, __global unsigned *c,
         unsigned m, unsigned k, unsigned n) {
    /**достаем глобальные id**/
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    /**достаем локальные для рабочей группы id**/
    const unsigned local_x = get_local_id(0);
    const unsigned local_y = get_local_id(1);
    /**позиция в одномерном массиве**/
    const unsigned position = local_y * TILE_SIZE + local_x;

    /**инициализируем локальную память. Будем выгружать туда квадратики из a и из b, а затем будем
     * эти квадратики перемножать, формируя квадратик ответа.
    **/
    __local unsigned buffer_a[TILE_SIZE * TILE_SIZE];
    __local unsigned buffer_b[TILE_SIZE * TILE_SIZE];

    /**постепенно накапливаемый результат**/
    unsigned result = 0;
    for (unsigned i = 0; i < (k + TILE_SIZE - 1) / TILE_SIZE; i++) {
        /**Вычисляем для каждого потока в рабочей группе, какой элемент нужно ему подгрузить в локальную память**/
        const unsigned shift = i * TILE_SIZE;
        /**подгружаем данные из глобальной памяти, если вылезли за границы, ничего страшного, заполним нулями**/
        buffer_a[position] = (shift + local_x < k && y < m) ? a[y * k + (shift + local_x)] : 0;
        buffer_b[position] = (x < n && shift + local_y < k) ? b[(shift + local_y) * n + x] : 0;

        /**дожидаемся всех потоков**/
        barrier(CLK_LOCAL_MEM_FENCE);

        /**каждый поток считает в конечном итоге свою ячейку в итоговом квадратике**/
        for (unsigned j = 0; j < TILE_SIZE; j++) {
            result = result || (buffer_a[local_y * TILE_SIZE + j] && buffer_b[j * TILE_SIZE + local_x]);
        }

        /**дожидаемся всех потоков**/
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    /**записываем результат в глобальную память**/
    if (x < n && y < m) {
        c[y * n + x] = result;
    }
}
