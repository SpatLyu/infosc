#include <vector>
#include "InfoTheory.hpp"
// 'Rcpp.h' should not be included and correct to include only 'RcppArmadillo.h'.
// #include <Rcpp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double RcppDiscEntropy(Rcpp::RObject vec, double base = 2) {
  if (TYPEOF(vec) == REALSXP) {
    std::vector<double> input = Rcpp::as<std::vector<double>>(vec);
    return DiscEntropy(input, base);
  } else if (TYPEOF(vec) == INTSXP) {
    std::vector<int> input = Rcpp::as<std::vector<int>>(vec);
    return DiscEntropy(input, base);
  } else {
    Rcpp::stop("Unsupported vector type. Must be numeric or integer.");
  }
}
