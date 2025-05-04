#include <vector>
#include "InfoTheory.hpp"
// 'Rcpp.h' should not be included and correct to include only 'RcppArmadillo.h'.
// #include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double RcppDiscEntropy(const Rcpp::NumericVector& vec, double base = 2) {
  std::vector<double> vec_std = Rcpp::as<std::vector<double>>(vec);
  return DiscEntropy(vec_std, base);
}

// [[Rcpp::export]]
double RcppDiscJoinEntropy(const Rcpp::NumericMatrix& mat,
                           const Rcpp::IntegerVector& columns,
                           double base = 2) {
  int numRows = mat.nrow();
  int numCols = mat.ncol();
  // Convert Rcpp NumericMatrix to std::vector<std::vector<double>>
  std::vector<std::vector<double>> cppMat(numRows, std::vector<double>(numCols));
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numCols; ++j) {
      cppMat[i][j] = mat(i, j);
    }
  }

  // Convert Rcpp IntegerVector to std::vector<int>
  std::vector<int> col_std = Rcpp::as<std::vector<int>>(columns);
  for (size_t i = 0; i < col_std.size(); ++i) {
    if (col_std[i] < 1 || col_std[i] > numCols) {
      Rcpp::stop("Each index in 'columns' must be between 1 and %d (inclusive).", numCols);
    }
    col_std[i] -= 1;
  }
  return DiscJoinEntropy(cppMat,col_std,base);
}
