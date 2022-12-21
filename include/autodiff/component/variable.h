#ifndef ADGC_AUTODIFF_VARIABLE_H_
#define ADGC_AUTODIFF_VARIABLE_H_

#include "node.h"

#include <string>
#include <vector>

namespace auto_diff {

class Variable : public Node {
public:
  Variable();
  Variable(const tensor::TensorShape &shape);
  Variable(const tensor::TensorShape &shape, Graph *graph);
  Variable(const tensor::TensorShape &shape, const bool &trainable,
           Graph *graph);
  Variable(const tensor::TensorShape &shape, const std::vector<Node *> &parents,
           Graph *graph);
  Variable(const tensor::TensorShape &shape, const std::vector<Node *> &parents,
           const std::string &name = "", const bool &random_init = true,
           const bool &trainable = true, Graph *graph = nullptr);

  inline void set_trainable(bool is_trainable) { trainable_ = is_trainable; };
  inline bool is_trainable() { return trainable_; };

protected:
  bool trainable_;

  void do_forward() override{}; // do nothing
  DTensor do_backward(Node *parent) override;
};

class Parameter : public Node {
public:
  Parameter();
  Parameter(const tensor::TensorShape &shape, const std::string &name = "",
            Graph *graph = nullptr);

  inline void set_trainable(bool is_trainable) { trainable_ = is_trainable; };
  inline bool is_trainable() { return trainable_; };

protected:
  bool trainable_;

  void do_forward() override{}; // do nothing
  DTensor do_backward(Node *parent) override;
};

} // namespace auto_diff

#endif