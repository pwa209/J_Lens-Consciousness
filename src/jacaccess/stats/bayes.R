args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: Rscript bayes.R participant-contrasts.csv bayes-factor.json")
}
if (!requireNamespace("BayesFactor", quietly = TRUE)) {
  stop("BayesFactor is not installed; run environment/install_r_packages.R")
}
values <- read.csv(args[[1]])$contrast
values <- values[is.finite(values)]
if (length(values) < 2) stop("at least two finite contrasts are required")
bf <- BayesFactor::ttestBF(x = values, mu = 0, nullInterval = c(0, Inf), rscale = 0.5)
result <- list(
  participants = length(values),
  mean_contrast = mean(values),
  directional_bayes_factor = as.numeric(bf[1]),
  rscale = 0.5
)
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is not installed")
}
jsonlite::write_json(result, args[[2]], auto_unbox = TRUE, pretty = TRUE)
