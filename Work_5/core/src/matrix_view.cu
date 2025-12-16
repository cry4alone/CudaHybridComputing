#include "../includes/matrix_view.cuh"
#include "../includes/matrix.cuh"

MatrixView Matrix::view() const {
  return {data_->device_data(), rows_, cols_};
}