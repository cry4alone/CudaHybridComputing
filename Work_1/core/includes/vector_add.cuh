#ifndef WORK1_VECTOR_ADD_H
#define WORK1_VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Ядро CUDA для поэлементного сложения двух векторов: C = A + B
 *
 * Выполняется на GPU. Каждый поток обрабатывает один элемент вектора.
 * Предполагается, что размер вектора `n` не превышает общее число запущенных потоков.
 *
 * @param a Указатель на первый входной вектор (на GPU)
 * @param b Указатель на второй входной вектор (на GPU)
 * @param c Указатель на выходной вектор (на GPU)
 * @param n Количество элементов в векторах
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n);

/**
 * @brief Складывает два вектора на GPU: C = A + B
 *
 * Эта функция копирует данные на GPU, запускает ядро сложения
 * и возвращает результат в выходной массив.
 *
 * @param a Указатель на первый вектор (на CPU)
 * @param b Указатель на второй вектор (на CPU)
 * @param c Указатель на результат (на CPU)
 * @param n Количество элементов
 * @return 0 при успехе, -1 при ошибке
 */
int vector_add(const float* a, const float* b, float* c, int n);

/**
 * @brief Складывает два вектора на CPU: C = A + B
 *
 * Поэлементно складывает два массива чисел с плавающей точкой.
 * Выполняется на центральном процессоре.
 *
 * @param a Указатель на первый входной вектор
 * @param b Указатель на второй входной вектор
 * @param c Указатель на выходной вектор
 * @param n Количество элементов
 */
void vector_add_cpu(const float* a, const float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif // WORK1_VECTOR_ADD_H