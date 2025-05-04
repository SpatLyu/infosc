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

// [[Rcpp::export]]
double RcppDiscMI(const Rcpp::NumericMatrix& mat,
                  const Rcpp::IntegerVector& columns1,
                  const Rcpp::IntegerVector& columns2,
                  double base = 10){
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
  std::vector<int> col1 = Rcpp::as<std::vector<int>>(columns1);
  for (size_t i = 0; i < col1.size(); ++i) {
    if (col1[i] < 1 || col1[i] > numCols) {
      Rcpp::stop("Each index in 'columns1' must be between 1 and %d (inclusive).", numCols);
    }
    col1[i] -= 1;
  }
  std::vector<int> col2 = Rcpp::as<std::vector<int>>(columns2);
  for (size_t i = 0; i < col2.size(); ++i) {
    if (col2[i] < 1 || col2[i] > numCols) {
      Rcpp::stop("Each index in 'columns2' must be between 1 and %d (inclusive).", numCols);
    }
    col2[i] -= 1;
  }

  return columns(cppMat,col1,col2,base);
}

// [[Rcpp::export]]
double DiscCE(const Rcpp::NumericMatrix& mat,
              const Rcpp::IntegerVector& target_columns,
              const Rcpp::IntegerVector& conditional_columns,
              double base = 10){
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
  std::vector<int> col1 = Rcpp::as<std::vector<int>>(target_columns);
  for (size_t i = 0; i < col1.size(); ++i) {
    if (col1[i] < 1 || col1[i] > numCols) {
      Rcpp::stop("Each index in 'target_columns' must be between 1 and %d (inclusive).", numCols);
    }
    col1[i] -= 1;
  }
  std::vector<int> col2 = Rcpp::as<std::vector<int>>(conditional_columns);
  for (size_t i = 0; i < col2.size(); ++i) {
    if (col2[i] < 1 || col2[i] > numCols) {
      Rcpp::stop("Each index in 'conditional_columns' must be between 1 and %d (inclusive).", numCols);
    }
    col2[i] -= 1;
  }

  return DiscCE(cppMat,col1,col2,base);
}
