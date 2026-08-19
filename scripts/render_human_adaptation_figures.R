#!/usr/bin/env Rscript

# AAAS-oriented visual system for Experiment 2.
# Uses base R only so the production renderer has no package dependency.

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1L]]
}

mock_mode <- "--mock" %in% args
experiment_root <- arg_value("--experiment-root", "results-extension/human-adaptation")
output_root <- arg_value("--output", file.path(experiment_root, "figures-r"))
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

palette <- c(
  feedforward = "#184D77",
  private_modules = "#497987",
  recurrent = "#858D7E",
  shared_workspace = "#D48448",
  unlimited_shared_state = "#BE4A36",
  sham = "#E1BD89"
)
architectures <- names(palette)[1:5]
architecture_labels <- c(
  feedforward = "Feedforward",
  private_modules = "Private modules",
  recurrent = "Recurrent",
  shared_workspace = "Constrained shared",
  unlimited_shared_state = "Unlimited shared"
)
stage_labels <- c(
  random_init = "Random init",
  task_trained = "Task trained",
  human_adapted = "Human adapted",
  sham_adapted = "Sham adapted"
)

alpha <- function(colour, opacity) adjustcolor(colour, alpha.f = opacity)
se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(is.finite(x)))
mean_ci <- function(x) c(mean(x, na.rm = TRUE), 1.96 * se(x))
unpack_summary <- function(value) {
  if (is.matrix(value)) return(value)
  do.call(rbind, value)
}

make_mock_data <- function() {
  set.seed(20260816)
  stages <- expand.grid(
    architecture = architectures,
    seed = 0:19,
    stage = names(stage_labels),
    stringsAsFactors = FALSE
  )
  base <- c(feedforward = 1.16, private_modules = 1.34, recurrent = 1.27,
            shared_workspace = 1.40, unlimited_shared_state = 1.38)
  effects <- c(random_init = 0.30, task_trained = 0,
               human_adapted = -0.14, sham_adapted = -0.04)
  seed_noise <- rnorm(20, 0, 0.055)
  stages$rms_distance <- mapply(
    function(a, s, st) base[[a]] + effects[[st]] + seed_noise[s + 1] + rnorm(1, 0, 0.025),
    stages$architecture, stages$seed, stages$stage
  )
  stages$cosine_similarity <- pmin(0.95, pmax(-0.2, 1.05 - 0.48 * stages$rms_distance +
                                                rnorm(nrow(stages), 0, 0.025)))

  sham <- expand.grid(
    architecture = architectures,
    seed = 0:19,
    outer_fold = 0:4,
    stringsAsFactors = FALSE
  )
  specificity <- c(feedforward = 0.13, private_modules = 0.08, recurrent = 0.16,
                   shared_workspace = 0.12, unlimited_shared_state = 0.06)
  sham$alignment_gain <- specificity[sham$architecture] + rnorm(nrow(sham), 0, 0.055)
  sham$sham_gain <- 0.035 + rnorm(nrow(sham), 0, 0.03)
  sham$human_specific_gain <- sham$alignment_gain - sham$sham_gain
  sham$distance_task_trained <- base[sham$architecture] + rnorm(nrow(sham), 0, 0.04)
  sham$distance_human_adapted <- sham$distance_task_trained - sham$alignment_gain
  sham$distance_sham_adapted <- sham$distance_task_trained - sham$sham_gain

  costs <- sham[, c("architecture", "seed", "outer_fold")]
  costs$condition <- "human_adapted"
  displacement <- c(feedforward = 0.012, private_modules = 0.019, recurrent = 0.016,
                    shared_workspace = 0.022, unlimited_shared_state = 0.025)
  costs$relative_l2_parameter_displacement <- pmax(
    0.001, displacement[costs$architecture] + rnorm(nrow(costs), 0, 0.003)
  )
  costs$accuracy_change <- rnorm(nrow(costs), -0.001, 0.004)
  costs$performance_gate_passed <- abs(costs$accuracy_change) <= 0.02

  intervention <- expand.grid(
    architecture = architectures,
    seed = 0:19,
    stage = c("task_trained", "human_adapted"),
    stringsAsFactors = FALSE
  )
  causal_base <- c(feedforward = 0.055, private_modules = 0.035, recurrent = 0.065,
                  shared_workspace = 0.075, unlimited_shared_state = 0.045)
  causal_shift <- c(feedforward = 0.018, private_modules = 0.008, recurrent = 0.025,
                   shared_workspace = 0.021, unlimited_shared_state = 0.006)
  intervention$causal_specificity <- causal_base[intervention$architecture] +
    ifelse(intervention$stage == "human_adapted",
           causal_shift[intervention$architecture], 0) +
    rnorm(nrow(intervention), 0, 0.012)

  transfer <- expand.grid(
    architecture = architectures,
    seed = 0:19,
    outer_fold = 0:4,
    evaluation_contrast = c("Kronemer early", "Kronemer late", "Somato early", "Somato late"),
    stage = c("task_trained", "human_adapted"),
    stringsAsFactors = FALSE
  )
  contrast_effect <- c("Kronemer early" = 0.08, "Kronemer late" = 0.03,
                       "Somato early" = 0.06, "Somato late" = -0.01)
  transfer$rms_distance <- 1.35 + rnorm(nrow(transfer), 0, 0.04) -
    ifelse(transfer$stage == "human_adapted",
           contrast_effect[transfer$evaluation_contrast] +
             specificity[transfer$architecture] / 4, 0)

  metrics <- expand.grid(
    architecture = architectures,
    metric = c("Gain", "Broadcast", "Persistence", "Concentration"),
    stringsAsFactors = FALSE
  )
  metrics$improvement <- c(
    0.24, 0.10, 0.16, 0.20, 0.08,
    0.18, 0.07, 0.14, 0.22, 0.10,
    0.12, 0.06, 0.19, 0.16, 0.05,
    0.20, 0.09, 0.11, 0.17, 0.07
  ) + rnorm(nrow(metrics), 0, 0.025)
  list(stages = stages, sham = sham, costs = costs,
       intervention = intervention, transfer = transfer, metrics = metrics)
}

read_real_data <- function() {
  aggregate_root <- file.path(experiment_root, "aggregate")
  required <- c(
    "seed-stage-summary.csv", "sham-comparison.csv", "adaptation-cost.csv",
    "post-adaptation-interventions.csv", "external-transfer.csv", "stage-geometry.csv"
  )
  missing <- required[!file.exists(file.path(aggregate_root, required))]
  if (length(missing)) stop("Missing production inputs: ", paste(missing, collapse = ", "))
  stages <- read.csv(file.path(aggregate_root, "seed-stage-summary.csv"))
  sham <- read.csv(file.path(aggregate_root, "sham-comparison.csv"))
  costs <- read.csv(file.path(aggregate_root, "adaptation-cost.csv"))
  intervention <- read.csv(file.path(aggregate_root, "post-adaptation-interventions.csv"))
  transfer <- read.csv(file.path(aggregate_root, "external-transfer.csv"))
  geometry <- read.csv(file.path(aggregate_root, "stage-geometry.csv"))
  geometry$absolute_residual <- abs(geometry$standardized_residual)
  wide <- reshape(
    aggregate(absolute_residual ~ architecture + seed + outer_fold + metric + stage,
              geometry, mean),
    idvar = c("architecture", "seed", "outer_fold", "metric"),
    timevar = "stage", direction = "wide"
  )
  wide$improvement <- wide$absolute_residual.task_trained -
    wide$absolute_residual.human_adapted
  metrics <- aggregate(improvement ~ architecture + metric, wide, mean)
  metrics$metric <- tools::toTitleCase(metrics$metric)
  list(stages = stages, sham = sham, costs = costs,
       intervention = intervention, transfer = transfer, metrics = metrics)
}

data <- if (mock_mode) make_mock_data() else read_real_data()

panel_label <- function(label) {
  mtext(label, side = 3, adj = -0.08, line = 0.4, font = 2, cex = 1.05)
}

plot_schematic <- function() {
  par(mar = c(0.4, 0.4, 1.5, 0.4), xaxs = "i", yaxs = "i")
  plot.new(); plot.window(xlim = c(0, 1), ylim = c(0, 1))
  text(0.003, 0.98, "a", adj = c(0, 1), font = 2, cex = 1.05)
  title("Human-neural adaptation is trained upstream and evaluated out-of-sample",
        adj = 0, cex.main = 0.95, font.main = 2)
  box_node <- function(x, y, w, h, text, fill, border = fill, text_col = "#202020") {
    rect(x, y, x + w, y + h, col = fill, border = border, lwd = 1.1)
    text(x + w / 2, y + h / 2, text, cex = 0.70, col = text_col)
  }
  arrow <- function(x0, y0, x1, y1, col = "#5E6468", lty = 1) {
    arrows(x0, y0, x1, y1, length = 0.055, angle = 22, col = col, lwd = 1.2, lty = lty)
  }
  box_node(0.01, 0.52, 0.14, 0.27, "Gabor EEG\ntrain subjects", alpha(palette[["feedforward"]], .12), palette[["feedforward"]])
  box_node(0.19, 0.52, 0.18, 0.27, "Temporal neural target\n6 x 6 activity RSM\n(mean + variance)", "#F6F4EF", "#7B7B76")
  box_node(0.43, 0.58, 0.13, 0.20, "Task-trained\ncheckpoint", alpha(palette[["private_modules"]], .13), palette[["private_modules"]])
  box_node(0.62, 0.68, 0.14, 0.20, "Human\nadapted", alpha(palette[["shared_workspace"]], .18), palette[["shared_workspace"]])
  box_node(0.62, 0.40, 0.14, 0.20, "Matched\nsham", alpha(palette[["sham"]], .48), palette[["shared_workspace"]])
  box_node(0.82, 0.52, 0.17, 0.27, "Held-out subjects\n4-metric geometry\n+ transfer + causality", "#F4F4F2", "#555555")
  arrow(0.15, 0.655, 0.19, 0.655, palette[["feedforward"]])
  arrow(0.37, 0.655, 0.43, 0.68, palette[["feedforward"]])
  arrow(0.56, 0.68, 0.62, 0.78, palette[["shared_workspace"]])
  arrow(0.56, 0.66, 0.62, 0.50, palette[["shared_workspace"]])
  arrow(0.76, 0.78, 0.82, 0.68, "#555555")
  arrow(0.76, 0.50, 0.82, 0.62, "#555555")
  rect(0.17, 0.08, 0.79, 0.25, col = "#FAF7F1", border = NA)
  text(0.48, 0.205,
       "Evaluation-only: gain  |  broadcast  |  persistence  |  concentration  |  RMS/cosine",
       cex = 0.70, col = "#343434")
  text(0.48, 0.125,
       "None enters adaptation loss or checkpoint selection",
       cex = 0.72, font = 2, col = palette[["unlimited_shared_state"]])
  segments(0.28, 0.29, 0.28, 0.38, lty = 3, col = palette[["unlimited_shared_state"]])
  segments(0.28, 0.38, 0.84, 0.38, lty = 3, col = palette[["unlimited_shared_state"]])
}

plot_stage_forest <- function() {
  par(mar = c(3.6, 7.0, 2.2, 0.7))
  grouped <- aggregate(rms_distance ~ architecture + stage, data$stages,
                       function(x) c(mean = mean(x), ci = 1.96 * se(x)))
  values <- unpack_summary(grouped$rms_distance)
  grouped$mean <- values[, 1]; grouped$ci <- values[, 2]
  xlim <- range(grouped$mean + c(-1, 1) * grouped$ci) + c(-0.08, 0.08)
  plot(NA, xlim = xlim, ylim = c(0.5, 5.5), yaxt = "n",
       xlab = "Held-out RMS distance (lower is closer)", ylab = "", bty = "n")
  panel_label("b")
  title("Stage geometry", adj = 0.06, cex.main = 0.9)
  row_labels <- paste0(LETTERS[1:5], "  ", architecture_labels[architectures])
  axis(2, at = 5:1, labels = row_labels, las = 1, tick = FALSE, cex.axis = 0.72)
  abline(h = 1:5, col = "#ECEAE6", lwd = 0.8)
  offsets <- c(random_init = 0.21, task_trained = 0.07, human_adapted = -0.07, sham_adapted = -0.21)
  shapes <- c(random_init = 1, task_trained = 16, human_adapted = 17, sham_adapted = 15)
  stage_cols <- c(random_init = "#858D7E", task_trained = "#184D77",
                  human_adapted = "#BE4A36", sham_adapted = "#E1BD89")
  for (i in seq_along(architectures)) {
    a <- architectures[[i]]; y0 <- 6 - i
    subset <- grouped[grouped$architecture == a, ]
    task_x <- subset$mean[subset$stage == "task_trained"]
    human_x <- subset$mean[subset$stage == "human_adapted"]
    segments(task_x, y0 + offsets[["task_trained"]], human_x,
             y0 + offsets[["human_adapted"]], col = alpha(palette[[a]], .55), lwd = 1.2)
    for (st in names(offsets)) {
      row <- subset[subset$stage == st, ]
      arrows(row$mean - row$ci, y0 + offsets[[st]], row$mean + row$ci,
             y0 + offsets[[st]], angle = 90, code = 3, length = 0.025,
             col = stage_cols[[st]], lwd = 0.9)
      points(row$mean, y0 + offsets[[st]], pch = shapes[[st]],
             col = stage_cols[[st]], bg = stage_cols[[st]], cex = 0.72)
    }
  }
  legend("top", legend = stage_labels, pch = shapes, col = stage_cols,
         pt.bg = stage_cols, bty = "n", horiz = TRUE,
         cex = 0.52, inset = c(0, 0.01))
}

plot_specific_gain <- function() {
  par(mar = c(3.6, 2.7, 2.2, 0.7))
  seed <- aggregate(human_specific_gain ~ architecture + seed, data$sham, mean)
  summary <- aggregate(human_specific_gain ~ architecture, seed,
                       function(x) c(mean = mean(x), ci = 1.96 * se(x)))
  vals <- unpack_summary(summary$human_specific_gain)
  summary$mean <- vals[, 1]; summary$ci <- vals[, 2]
  summary <- summary[match(architectures, summary$architecture), ]
  lim <- range(c(0, summary$mean - summary$ci, summary$mean + summary$ci)) + c(-0.025, 0.025)
  plot(NA, xlim = lim, ylim = c(0.5, 5.5), yaxt = "n",
       xlab = "Human-specific gain\n(sham distance - adapted distance)", ylab = "", bty = "n")
  panel_label("c")
  title("Specificity over sham", adj = 0.15, cex.main = 0.9)
  abline(v = 0, col = "#777777", lty = 3)
  for (i in seq_along(architectures)) {
    a <- architectures[[i]]; y <- 6 - i
    values <- seed$human_specific_gain[seed$architecture == a]
    points(values, y + seq(-0.13, 0.13, length.out = length(values)),
           pch = 16, cex = 0.34, col = alpha(palette[[a]], .30))
    arrows(summary$mean[i] - summary$ci[i], y, summary$mean[i] + summary$ci[i], y,
           angle = 90, code = 3, length = 0.035, lwd = 1.1, col = palette[[a]])
    points(summary$mean[i], y, pch = 18, cex = 1.0, col = palette[[a]])
  }
  text(par("usr")[1], 5:1, labels = LETTERS[1:5], pos = 4, cex = 0.58, col = "#666666")
}

plot_metric_heatmap <- function() {
  par(mar = c(5.7, 5.0, 2.2, 1.1))
  metrics <- c("Gain", "Broadcast", "Persistence", "Concentration")
  matrix <- outer(architectures, metrics, Vectorize(function(a, m) {
    value <- data$metrics$improvement[data$metrics$architecture == a & data$metrics$metric == m]
    ifelse(length(value), value[[1]], NA_real_)
  }))
  matrix <- matrix[rev(seq_len(nrow(matrix))), , drop = FALSE]
  limit <- max(abs(matrix), na.rm = TRUE)
  ramp <- colorRampPalette(c(palette[["feedforward"]], "#F7F5F0", palette[["unlimited_shared_state"]]))(101)
  image(seq_along(metrics), seq_along(architectures), t(matrix), col = ramp,
        zlim = c(-limit, limit), axes = FALSE, xlab = "", ylab = "")
  panel_label("d")
  title("Metric fingerprint", adj = 0.24, cex.main = 0.9)
  axis(1, at = seq_along(metrics), labels = metrics, las = 2, tick = FALSE, cex.axis = 0.64)
  axis(2, at = seq_along(architectures), labels = rev(LETTERS[1:5]), las = 1, tick = FALSE, cex.axis = 0.6)
  for (i in seq_along(metrics)) for (j in seq_along(architectures)) {
    text(i, j, sprintf("%+.2f", matrix[j, i]), cex = 0.58,
         col = if (abs(matrix[j, i]) > .65 * limit) "white" else "#262626")
  }
  mtext("Closer after adaptation  ->", side = 1, line = 4.5, cex = 0.61, col = "#555555")
}

plot_cost_gain <- function() {
  par(mar = c(4.2, 4.2, 2.2, 0.8))
  gain <- aggregate(alignment_gain ~ architecture + seed, data$sham, mean)
  cost <- aggregate(relative_l2_parameter_displacement ~ architecture + seed,
                    data$costs[data$costs$condition == "human_adapted", ], mean)
  merged <- merge(gain, cost, by = c("architecture", "seed"))
  xlim <- range(merged$relative_l2_parameter_displacement) * c(.86, 1.1)
  ylim <- range(c(0, merged$alignment_gain)) + c(-.02, .03)
  plot(NA, xlim = xlim, ylim = ylim, bty = "n",
       xlab = "Relative parameter displacement", ylab = "Held-out alignment gain")
  panel_label("a")
  title("Adaptation efficiency", adj = 0, cex.main = 0.9)
  abline(h = 0, col = "#777777", lty = 3)
  for (a in architectures) {
    subset <- merged[merged$architecture == a, ]
    points(subset$relative_l2_parameter_displacement, subset$alignment_gain,
           pch = 16, cex = .52, col = alpha(palette[[a]], .45))
    center <- c(mean(subset$relative_l2_parameter_displacement), mean(subset$alignment_gain))
    points(center[1], center[2], pch = 21, bg = palette[[a]], col = "white", lwd = .8, cex = 1.25)
  }
  legend("topright", architecture_labels[architectures], pch = 16,
         col = palette[architectures], bty = "n", cex = .62)
  usr <- par("usr")
  text(usr[1] + .025 * diff(usr[1:2]), usr[4] - .035 * diff(usr[3:4]),
       "preferred", pos = 4, cex = .58, col = "#666666")
}

plot_retention <- function() {
  par(mar = c(4.2, 7.0, 2.2, 0.7))
  costs <- data$costs[data$costs$condition == "human_adapted", ]
  seed <- aggregate(accuracy_change ~ architecture + seed, costs, mean)
  summary <- aggregate(accuracy_change ~ architecture, seed,
                       function(x) c(mean = mean(x), ci = 1.96 * se(x)))
  vals <- unpack_summary(summary$accuracy_change)
  summary$mean <- vals[, 1]; summary$ci <- vals[, 2]
  summary <- summary[match(architectures, summary$architecture), ]
  plot(NA, xlim = c(-.026, .026), ylim = c(.5, 5.5), yaxt = "n", bty = "n",
       xlab = "Presence accuracy change", ylab = "")
  panel_label("b")
  title("Task retention", adj = 0, cex.main = .9)
  rect(-.02, .5, .02, 5.5, col = alpha(palette[["sham"]], .28), border = NA)
  abline(v = c(-.02, 0, .02), col = c("#777777", "#444444", "#777777"),
         lty = c(3, 1, 3), lwd = c(.8, .8, .8))
  axis(2, 5:1, architecture_labels[architectures], las = 1, tick = FALSE, cex.axis = .69)
  for (i in seq_along(architectures)) {
    a <- architectures[[i]]; y <- 6 - i
    arrows(summary$mean[i] - summary$ci[i], y, summary$mean[i] + summary$ci[i], y,
           angle = 90, code = 3, length = .035, col = palette[[a]], lwd = 1.2)
    points(summary$mean[i], y, pch = 18, col = palette[[a]], cex = 1.0)
  }
  text(.019, .72, "prespecified gate", pos = 2, cex = .58, col = "#666666")
}

plot_causality <- function() {
  par(mar = c(4.2, 4.2, 2.2, .8))
  intervention <- aggregate(causal_specificity ~ architecture + seed + stage,
                            data$intervention, mean)
  summary <- aggregate(causal_specificity ~ architecture + stage, intervention,
                       function(x) c(mean = mean(x), ci = 1.96 * se(x)))
  vals <- unpack_summary(summary$causal_specificity)
  summary$mean <- vals[, 1]; summary$ci <- vals[, 2]
  ylim <- range(c(0, summary$mean - summary$ci, summary$mean + summary$ci)) + c(-.01, .015)
  plot(NA, xlim = c(.65, 5.35), ylim = ylim, xaxt = "n", bty = "n",
       xlab = "", ylab = "Targeted drop - random drop")
  panel_label("c")
  title("Causal specificity", adj = 0, cex.main = .9)
  axis(1, 1:5, LETTERS[1:5], tick = FALSE, cex.axis = .68)
  abline(h = 0, col = "#777777", lty = 3)
  for (i in seq_along(architectures)) {
    a <- architectures[[i]]
    before <- summary[summary$architecture == a & summary$stage == "task_trained", ]
    after <- summary[summary$architecture == a & summary$stage == "human_adapted", ]
    segments(i - .13, before$mean, i + .13, after$mean, col = alpha(palette[[a]], .65), lwd = 1.2)
    arrows(i - .13, before$mean - before$ci, i - .13, before$mean + before$ci,
           angle = 90, code = 3, length = .025, col = palette[[a]])
    arrows(i + .13, after$mean - after$ci, i + .13, after$mean + after$ci,
           angle = 90, code = 3, length = .025, col = palette[[a]])
    points(i - .13, before$mean, pch = 1, col = palette[[a]], cex = .9)
    points(i + .13, after$mean, pch = 16, col = palette[[a]], cex = .8)
  }
  legend("topleft", c("Task trained", "Human adapted"), pch = c(1, 16),
         col = "#4A4A4A", bty = "n", cex = .62)
}

plot_transfer <- function() {
  par(mar = c(5.5, 7.0, 2.2, 1.0))
  transfer <- aggregate(rms_distance ~ architecture + seed + outer_fold +
                          evaluation_contrast + stage, data$transfer, mean)
  wide <- reshape(transfer, idvar = c("architecture", "seed", "outer_fold", "evaluation_contrast"),
                  timevar = "stage", direction = "wide")
  wide$gain <- wide$rms_distance.task_trained - wide$rms_distance.human_adapted
  summary <- aggregate(gain ~ architecture + evaluation_contrast, wide, mean)
  contrasts <- unique(as.character(summary$evaluation_contrast))
  matrix <- outer(architectures, contrasts, Vectorize(function(a, c) {
    summary$gain[summary$architecture == a & summary$evaluation_contrast == c][1]
  }))
  matrix <- matrix[rev(seq_len(nrow(matrix))), , drop = FALSE]
  limit <- max(abs(matrix), na.rm = TRUE)
  ramp <- colorRampPalette(c(palette[["feedforward"]], "#F7F5F0", palette[["unlimited_shared_state"]]))(101)
  image(seq_along(contrasts), seq_along(architectures), t(matrix), col = ramp,
        zlim = c(-limit, limit), axes = FALSE, xlab = "", ylab = "")
  panel_label("d")
  title("Transfer beyond Gabor", adj = 0, cex.main = .9)
  axis(1, seq_along(contrasts), contrasts, las = 2, tick = FALSE, cex.axis = .63)
  axis(2, seq_along(architectures), rev(LETTERS[1:5]),
       las = 1, tick = FALSE, cex.axis = .66)
  for (i in seq_along(contrasts)) for (j in seq_along(architectures)) {
    text(i, j, sprintf("%+.2f", matrix[j, i]), cex = .60,
         col = if (abs(matrix[j, i]) > .65 * limit) "white" else "#262626")
  }
  mtext("Positive values indicate cross-dataset convergence", side = 1, line = 4.4,
        cex = .60, col = "#555555")
}

draw_figure_1 <- function(mock_stamp = FALSE) {
  layout(matrix(c(1, 1, 1, 2, 3, 4), nrow = 2, byrow = TRUE),
         heights = c(.32, .68), widths = c(.47, .26, .27))
  par(family = "sans", mgp = c(2.0, .55, 0), tcl = -.18,
      cex.axis = .72, cex.lab = .78, col.axis = "#282828", col.lab = "#282828")
  plot_schematic(); plot_stage_forest(); plot_specific_gain(); plot_metric_heatmap()
  if (mock_stamp) mtext("MOCK DATA - LAYOUT ONLY", outer = TRUE, side = 4,
                        adj = .02, line = -0.7, cex = .54, col = "#8A8A85")
}

draw_figure_2 <- function(mock_stamp = FALSE) {
  layout(matrix(1:4, nrow = 2, byrow = TRUE), widths = c(.48, .52), heights = c(.5, .5))
  par(family = "sans", mgp = c(2.0, .55, 0), tcl = -.18,
      cex.axis = .72, cex.lab = .78, col.axis = "#282828", col.lab = "#282828")
  plot_cost_gain(); plot_retention(); plot_causality(); plot_transfer()
  if (mock_stamp) mtext("MOCK DATA - LAYOUT ONLY", outer = TRUE, side = 4,
                        adj = .02, line = -0.7, cex = .54, col = "#8A8A85")
}

render_png <- function(path, draw) {
  png(path, width = 2160, height = 1710, res = 300, bg = "white", type = "windows")
  par(oma = c(.45, .25, .20, .20))
  draw(mock_mode)
  dev.off()
}

render_pdf <- function(path, draw) {
  pdf(path, width = 7.2, height = 5.7, family = "Helvetica", useDingbats = FALSE,
      onefile = TRUE, paper = "special")
  par(oma = c(.45, .25, .20, .20))
  draw(mock_mode)
  dev.off()
}

suffix <- if (mock_mode) "mock" else "final"
fig1_base <- file.path(output_root, paste0("Fig7-human-adaptation-trajectory-", suffix))
fig2_base <- file.path(output_root, paste0("Fig8-adaptation-cost-causality-", suffix))
render_png(paste0(fig1_base, ".png"), draw_figure_1)
render_png(paste0(fig2_base, ".png"), draw_figure_2)
render_pdf(paste0(fig1_base, ".pdf"), draw_figure_1)
render_pdf(paste0(fig2_base, ".pdf"), draw_figure_2)

if (mock_mode) {
  combined <- file.path(output_root, "human-adaptation-figure-mockbook.pdf")
  pdf(combined, width = 7.2, height = 5.7, family = "Helvetica", useDingbats = FALSE,
      onefile = TRUE, paper = "special")
  par(oma = c(.45, .25, .20, .20)); draw_figure_1(TRUE)
  par(oma = c(.45, .25, .20, .20)); draw_figure_2(TRUE)
  dev.off()
}

cat("Rendered R figures to", normalizePath(output_root, winslash = "/", mustWork = TRUE), "\n")
