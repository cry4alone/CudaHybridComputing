#ifndef WORK1_VECTOR_ADD_H
#define WORK1_VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vector_add_kernel(float* a, float* b, float* c, int n);

/**
 * @brief Складывает два вектора на GPU: C = A + B
 *
 * Эта функция копирует данные на GPU, запускает ядро сложения
 * и возвращает результат в выходной массив.
 *
 * @param a Указатель на первый вектор
 * @param b Указатель на второй вектор
 * @param c Указатель на результат
 * @param n Количество элементов
 * @return 0 при успехе, -1 при ошибке
 */
int vector_add(const float* a, const float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

#endif // WORK1_VECTOR_ADD_H