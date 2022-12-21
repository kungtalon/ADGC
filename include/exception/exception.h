#ifndef ADGC_EXCEPTION_EXCEPTION_H_
#define ADGC_EXCEPTION_EXCEPTION_H_

#include <exception>

namespace adg_exception {

class AutoDiffGraphException : public std::exception {
public:
  AutoDiffGraphException(){};
  AutoDiffGraphException(const std::string &msg) : message_(msg){};
  const char *what() const noexcept { return message_.c_str(); };

protected:
  std::string message_;
};

// errors for tensor class
class TensorException : public AutoDiffGraphException {
public:
  TensorException() : AutoDiffGraphException("Tensor Exception"){};
  TensorException(const std::string &msg)
      : AutoDiffGraphException("Tensor Exception >> " + msg){};
};

class EmptyTensorError : public TensorException {
public:
  EmptyTensorError() : TensorException("EmptyTensorError"){};
};

class IndexOutOfRangeError : public TensorException {
public:
  IndexOutOfRangeError() : TensorException("IndexOutOfRangeError"){};
};

class AxisOutOfRangeError : public TensorException {
public:
  AxisOutOfRangeError() : TensorException("AxisOutOfRangeError"){};
  AxisOutOfRangeError(const std::string msg) : TensorException(msg){};
};

class MismatchTensorDimError : public TensorException {
public:
  MismatchTensorDimError() : TensorException("MismatchTensorDimError"){};
  MismatchTensorDimError(const std::string msg) : TensorException(msg){};
};

class MismatchTensorShapeError : public TensorException {
public:
  MismatchTensorShapeError() : TensorException("MismatchTensorShapeError"){};
  MismatchTensorShapeError(const std::string msg) : TensorException(msg){};
};

class MismatchTensorTypeError : public TensorException {
public:
  MismatchTensorTypeError() : TensorException("MismatchTensorTypeError"){};
};

class ResultTensorShapeError : public TensorException {
public:
  ResultTensorShapeError() : TensorException("ResultTensorShapeError"){};
};

class InvalidDtypeException : public TensorException {
public:
  InvalidDtypeException() : TensorException("InvalidDtypeException"){};
};

class InvalidTensorShapeException : public TensorException {
public:
  InvalidTensorShapeException()
      : TensorException("InvalidTensorShapeException"){};
  InvalidTensorShapeException(const std::string &msg) : TensorException(msg){};
};

class InvalidTensorIndexException : public TensorException {
public:
  InvalidTensorIndexException()
      : TensorException("InvalidTensorIndexException"){};
};

class NonImplementedException : public AutoDiffGraphException {
public:
  NonImplementedException()
      : AutoDiffGraphException("NonImplementedException"){};
};

class TestingDebugException : public AutoDiffGraphException {
public:
  TestingDebugException(){};
  TestingDebugException(const std::string &msg) : AutoDiffGraphException(msg){};
};

class NodeNotFoundError : public AutoDiffGraphException {
public:
  NodeNotFoundError(){};
  NodeNotFoundError(const std::string &msg) : AutoDiffGraphException(msg){};
};

class DuplicateNodeNameError : public AutoDiffGraphException {
public:
  DuplicateNodeNameError(){};
  DuplicateNodeNameError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class MismatchRegisterdGraphError : public AutoDiffGraphException {
public:
  MismatchRegisterdGraphError(){};
  MismatchRegisterdGraphError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class MismatchNodeValueShapeError : public AutoDiffGraphException {
public:
  MismatchNodeValueShapeError(){};
  MismatchNodeValueShapeError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class IncompatibleNodeValueShapeError : public AutoDiffGraphException {
public:
  IncompatibleNodeValueShapeError(){};
  IncompatibleNodeValueShapeError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class OpsParentsNumException : public AutoDiffGraphException {
public:
  OpsParentsNumException(){};
  OpsParentsNumException(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class OpsParentsUnsetException : public AutoDiffGraphException {
public:
  OpsParentsUnsetException(){};
  OpsParentsUnsetException(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class FunctionalParentsNumException : public AutoDiffGraphException {
public:
  FunctionalParentsNumException(){};
  FunctionalParentsNumException(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class FunctionalParentsUnsetException : public AutoDiffGraphException {
public:
  FunctionalParentsUnsetException(){};
  FunctionalParentsUnsetException(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class UnexpectedParentTypeError : public AutoDiffGraphException {
public:
  UnexpectedParentTypeError(){};
  UnexpectedParentTypeError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

class GradError : public AutoDiffGraphException {
public:
  GradError(){};
  GradError(const std::string &msg) : AutoDiffGraphException(msg){};
};

class InvalidNodeOperationError : public AutoDiffGraphException {
public:
  InvalidNodeOperationError(){};
  InvalidNodeOperationError(const std::string &msg)
      : AutoDiffGraphException(msg){};
};

} // namespace adg_exception

#endif