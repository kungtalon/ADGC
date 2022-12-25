//
// Created by kungtalon on 2022/12/23.
//

#include "tensor/tensor.h"
#include "data/dataset.h"
#include "autodiff/graph.h"
#include "autodiff/metric/metric.h"
#include "autodiff/layer/layer.h"
#include "autodiff/optimizer/optimizer.h"

using namespace auto_diff;

void train(const size_t &batch_size,
           const int &epoch,
           const std::string &train_data_path,
           const std::string &test_data_path) {
  // setup dataset
  auto train_data_set = data::CsvDataset(train_data_path, 0, batch_size, ",", true);
  auto test_data_set = data::CsvDataset(test_data_path, 0, batch_size, ",", true, false);
  DataPair paired_train_data, paired_test_data;

  size_t output_dim = train_data_set.get_label_count();

  std::cout << "[INFO] Dataset loaded!!!" << std::endl;

  // setup variables
  Variable x = Variable({batch_size, 784});
  Variable labels = Variable({batch_size, output_dim});

  // build graph
  layer::Dense dense_layer_1(784, 120, "relu");
  layer::Dense dense_layer_2(120, 24, "relu");
  layer::Dense dense_layer_3(24, output_dim, "none");

  auto loss = functional::cross_entropy_with_softmax(
    dense_layer_3(
      dense_layer_2(
        dense_layer_1(x))
    ), labels);

  auto optim = optimizer::Adam(loss, batch_size);

  std::cout << "[INFO] Started training!!!" << std::endl;

  int cur_epoch = 0, cur_step;
  double acc_loss = 0;
  while (cur_epoch++ < epoch) {
    cur_step = 0;
    while (train_data_set.has_next()) {
      paired_train_data = train_data_set.get_next();
      // forward
      x.assign_value(paired_train_data.first);
      labels.assign_value(paired_train_data.second);
      optim.zero_grad();
      loss.forward();

      // backward
      optim.step();

      if (cur_step && cur_step % 20 == 0) {
        std::cout << "[INFO] Cur Step: " << cur_step << "\tCur loss: " << acc_loss / 20 << std::endl;
        acc_loss = 0;
      }

      acc_loss += loss.get_value().get_value();
      ++cur_step;
    }

    train_data_set.reset_iterator();

    // evaluation
    double acc = 0;
    tensor::Tensor<double> preds;
    while (test_data_set.has_next()) {
      paired_test_data = test_data_set.get_next();
      x.assign_value(paired_test_data.first);
      labels.assign_value(paired_test_data.second);
      loss.forward();

      preds = loss.get_probs().arg_amax(1);
      acc += metric::accuracy(preds, labels.get_value(), false);
    }

    std::cout << "[INFO] Cur epoch: " << cur_epoch << " Cur accuracy: " << acc / test_data_set.get_data_count()
              << std::endl;
    test_data_set.reset_iterator();
  }

  Graph::clear_graph();
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    throw std::invalid_argument("Invalid number of arguments for training! 4 Needed...");
  }

  size_t batch_size;
  int epoch;
  std::string train_data_path, test_data_path;

  try {
    batch_size = std::stoi(argv[1]);
    epoch = std::stoi(argv[2]);
    train_data_path = argv[3];
    test_data_path = argv[4];
  } catch (const std::exception &ex) {
    throw std::runtime_error(std::string("Failed to convert the arguments... ") + ex.what());
  }

  train(batch_size, epoch, train_data_path, test_data_path);
  return 0;
}
