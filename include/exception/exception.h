#ifndef ADGC_EXCEPTION_EXCEPTION_H_
#define ADGC_EXCEPTION_EXCEPTION_H_

#include <exception>

namespace adg_exception {

class AutoDiffGraphException : public std::exception {};

// errors for tensor class
class TensorException : public AutoDiffGraphException {};

class EmptyTensorError : public TensorException {};

class IndexOutOfRangeError : public TensorException {};

class AxisOutOfRangeError : public TensorException {};

class MismatchTensorShapeError : public TensorException {};

class MismatchTensorTypeError : public TensorException {};

class ResultTensorShapeError : public TensorException {};

class InvalidDtypeException : public TensorException {};

class InvalidTensorShapeException : public TensorException {};

class InvalidTensorIndexException : public TensorException {};

class NonImplementedException : public AutoDiffGraphException {};

class TestingDebugException : public AutoDiffGraphException {};

class DuplicateNodeNameError : public AutoDiffGraphException {};

class MismatchRegisterdGraphError : public AutoDiffGraphException {};

} // namespace adg_exception

#endif