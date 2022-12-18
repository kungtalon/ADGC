#ifndef ADGC_EXCEPTION_EXCEPTION_H_
#define ADGC_EXCEPTION_EXCEPTION_H_

#include <exception>

namespace adg_exception {

class AutoDiffGraphException : public std::exception {
public:
  AutoDiffGraphException(){};
  AutoDiffGraphException(const std::string &msg) : message_(msg){};
  const char *what() { return message_.c_str(); };

protected:
  std::string message_;
};

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

class TestingDebugException : public AutoDiffGraphException {
public:
  TestingDebugException(){};
  TestingDebugException(const std::string &msg) : AutoDiffGraphException(msg){};
};

class DuplicateNodeNameError : public AutoDiffGraphException {};

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

} // namespace adg_exception

#endif