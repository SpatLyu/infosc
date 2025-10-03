#ifndef INFOTHEORY_HPP
#define INFOTHEORY_HPP

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <limits>
#include <sstream>

/**
 * Computes the entropy of a discrete sequence.
 * @tparam T Type of the input elements (e.g., int, double).
 * @param vec Input vector containing discrete values.
 * @param base Logarithm base (default: 2).
 * @return Entropy value.
 */
template <typename T>
double DiscEntropy(const std::vector<T>& vec, double base = 2) {
  std::unordered_map<T, int> counts;
  int n = vec.size();

  for (const T& x : vec) {
    counts[x]++;
  }

  if (n == 0) return std::numeric_limits<double>::quiet_NaN();

  const double log_base = std::log(base);
  double entropy = 0.0;

  for (const auto& pair : counts) {
    double p = static_cast<double>(pair.second) / n;
    entropy += p * std::log(p);
  }

  return -entropy / log_base;
}

/**
 * Computes the joint entropy of a multivariate discrete sequence.
 * @tparam T Type of the input elements (e.g., int, double).
 * @param mat Input matrix where each row represents a sample containing multiple variables.
 * @param columns The columns which are used in joint entropy estimation.
 * @param base Logarithm base (default: 2).
 * @return Joint entropy value.
 */
template <typename T>
double DiscJoinEntropy(const std::vector<std::vector<T>>& mat,
                       const std::vector<int>& columns,
                       double base = 2) {
  const double log_base = std::log(base);
  std::unordered_map<std::string, int> counts;
  int valid_count = 0;

  for (const auto& sample : mat) {
    std::ostringstream key_stream;
    for (size_t i = 0; i < columns.size(); ++i) {
      key_stream << sample[columns[i]] << "_";
    }
    counts[key_stream.str()]++;
    valid_count++;
  }

  if (valid_count == 0) return std::numeric_limits<double>::quiet_NaN();

  double entropy = 0.0;
  for (const auto& pair : counts) {
    double p = static_cast<double>(pair.second) / valid_count;
    entropy += p * std::log(p);
  }

  return -entropy / log_base;
}

/**
 * Computes the mutual information between two sets of discrete variables (columns).
 * @tparam T Type of the input elements (e.g., int, double).
 * @param mat Input matrix where each row represents a sample and each column a discrete variable.
 * @param columns1 Indices of columns representing the first set of variables (X).
 * @param columns2 Indices of columns representing the second set of variables (Y).
 * @param base Logarithm base used in entropy calculations (default: 2).
 * @return Mutual information value I(X; Y) = H(X) + H(Y) - H(X,Y).
 */
template <typename T>
double DiscMI(const std::vector<std::vector<T>>& mat,
              const std::vector<int>& columns1,
              const std::vector<int>& columns2,
              double base = 2) {
  std::unordered_set<int> unique_set;
  unique_set.insert(columns1.begin(), columns1.end());
  unique_set.insert(columns2.begin(), columns2.end());
  std::vector<int> columns(unique_set.begin(), unique_set.end());

  double h_x = DiscJoinEntropy(mat, columns1, base);
  double h_y = DiscJoinEntropy(mat, columns2, base);
  double h_xy = DiscJoinEntropy(mat, columns, base);

  if (std::isnan(h_x) || std::isnan(h_y) || std::isnan(h_xy)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return h_x + h_y - h_xy;
}

/**
 * Computes the conditional entropy H(X | Y) between two sets of discrete variables.
 * @tparam T Type of the input elements (e.g., int, double).
 * @param mat Input matrix where each row is a sample and each column is a discrete variable.
 * @param target_columns Indices of columns representing the target variable(s) X.
 * @param conditional_columns Indices of columns representing the conditioning variable(s) Y.
 * @param base Logarithm base used in entropy calculations (default: 2).
 * @return Conditional entropy value H(X | Y) = H(X,Y) - H(Y).
 */
template <typename T>
double DiscCE(const std::vector<std::vector<T>>& mat,
              const std::vector<int>& target_columns,
              const std::vector<int>& conditional_columns,
              double base = 2) {
  std::unordered_set<int> unique_set;
  unique_set.insert(target_columns.begin(), target_columns.end());
  unique_set.insert(conditional_columns.begin(), conditional_columns.end());
  std::vector<int> columns(unique_set.begin(), unique_set.end());

  double h_xy = DiscJoinEntropy(mat, columns, base);
  double h_y = DiscJoinEntropy(mat, conditional_columns, base);

  if (std::isnan(h_xy) || std::isnan(h_y)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  return h_xy - h_y;
}

#endif // INFOTHEORY_HPP
