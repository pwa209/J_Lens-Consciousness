#!/usr/bin/env Rscript

# Native-R six-figure system for the integrated Science Advances article.
# Statistical outputs are read from the renderer-neutral CSV layer created by
# prepare_science_advances_figure_data.py; this script performs no inferential
# tests and never changes content according to statistical significance.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(scales)
  library(jsonlite)
  library(digest)
})

args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = NULL) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1]]
}

data_dir <- value_after("--data")
output_dir <- value_after("--output")
manifest_path <- value_after("--manifest", file.path(output_dir, "manifest.json"))
if (is.null(data_dir) || is.null(output_dir)) {
  stop("Usage: Rscript render_science_advances_figures.R --data DIR --output DIR [--manifest FILE]")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

arch_order <- c(
  "feedforward", "recurrent", "private_modules",
  "shared_workspace", "unlimited_shared_state"
)
arch_labels <- c(
  feedforward = "Feedforward",
  recurrent = "Recurrent",
  private_modules = "Private modules",
  shared_workspace = "Capacity-limited shared",
  unlimited_shared_state = "Unlimited shared"
)
arch_short <- c(
  feedforward = "Feedforward",
  recurrent = "Recurrent",
  private_modules = "Private",
  shared_workspace = "Limited shared",
  unlimited_shared_state = "Unlimited shared"
)
pal <- c(
  feedforward = "#C22525",
  recurrent = "#3F3A39",
  private_modules = "#6F5E56",
  shared_workspace = "#C3AB8C",
  unlimited_shared_state = "#E1D6C7"
)
outline <- c(pal[1:4], unlimited_shared_state = "#6F5E56")
dataset_pal <- c(gabor = "#C22525", kronemer = "#C3AB8C", somato = "#6F5E56")
metric_order <- c("gain", "broadcast", "persistence", "concentration")
metric_labels <- c(
  gain = "Gain", broadcast = "Broadcast",
  persistence = "Persistence", concentration = "Concentration"
)
stage_order <- c("random_init", "task_trained", "human_adapted", "sham_adapted")
stage_labels <- c(
  random_init = "Random init.", task_trained = "Task trained",
  human_adapted = "Human adapted", sham_adapted = "Sham adapted"
)
ink <- "#3F3A39"
mid <- "#6F5E56"
sand <- "#C3AB8C"
paper <- "#E1D6C7"
red <- "#C22525"

read_fig <- function(name) {
  read_csv(file.path(data_dir, name), show_col_types = FALSE, progress = FALSE)
}

as_arch <- function(x) factor(x, levels = arch_order, labels = unname(arch_short[arch_order]))
as_metric <- function(x) factor(x, levels = metric_order, labels = unname(metric_labels[metric_order]))

theme_sa <- function(base_size = 8) {
  theme_classic(base_family = "Liberation Sans", base_size = base_size) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid.major.y = element_line(colour = alpha(paper, 0.72), size = 0.28),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = ink, size = 0.35),
      axis.ticks = element_line(colour = ink, size = 0.3),
      axis.ticks.length = grid::unit(1.4, "mm"),
      axis.title = element_text(size = rel(0.95), colour = ink),
      axis.text = element_text(size = rel(0.84), colour = ink),
      plot.title = element_text(face = "bold", size = rel(1.13), hjust = 0, margin = margin(b = 5, l = 14)),
      plot.subtitle = element_text(size = rel(0.84), colour = mid, margin = margin(b = 5)),
      plot.tag = element_text(face = "bold", size = rel(1.25), colour = ink),
      plot.tag.position = c(0.005, 0.995),
      legend.title = element_text(size = rel(0.78), colour = ink),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = rel(0.78)),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", colour = ink),
      plot.margin = margin(5, 6, 5, 6)
    )
}

theme_void_sa <- function() {
  theme_void(base_family = "Liberation Sans", base_size = 8) +
    theme(
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.title = element_text(face = "bold", size = 9, hjust = 0, margin = margin(b = 5, l = 14)),
      plot.tag = element_text(face = "bold", size = 10, colour = ink),
      plot.tag.position = c(0.015, 0.995),
      plot.margin = margin(5, 6, 5, 6)
    )
}

mean_ci <- function(x) {
  x <- x[is.finite(x)]
  m <- mean(x)
  h <- if (length(x) > 1) 1.96 * sd(x) / sqrt(length(x)) else 0
  tibble(mean = m, lower = m - h, upper = m + h, n = length(x))
}

div_scale <- function(limit = NULL, name = NULL) {
  scale_fill_gradientn(
    colours = c(ink, mid, paper, sand, red),
    values = rescale(c(-1, -0.45, 0, 0.45, 1)),
    limits = if (is.null(limit)) NULL else c(-limit, limit),
    oob = squish,
    name = name
  )
}

files_out <- list()
save_figure <- function(plot, stem, title, width = 7.2, height = 5.8) {
  targets <- c(png = file.path(output_dir, paste0(stem, ".png")),
               pdf = file.path(output_dir, paste0(stem, ".pdf")),
               svg = file.path(output_dir, paste0(stem, ".svg")))
  ggsave(targets[["png"]], plot, width = width, height = height, units = "in",
         dpi = 320, device = ragg::agg_png, bg = "white")
  ggsave(targets[["pdf"]], plot, width = width, height = height, units = "in",
         device = grDevices::cairo_pdf, bg = "white")
  ggsave(targets[["svg"]], plot, width = width, height = height, units = "in",
         device = svglite::svglite, bg = "white")
  for (fmt in names(targets)) {
    path <- targets[[fmt]]
    files_out[[length(files_out) + 1]] <<- list(
      path = normalizePath(path, winslash = "/", mustWork = TRUE),
      format = fmt,
      title = title,
      bytes = unname(file.info(path)$size),
      sha256 = digest(path, algo = "sha256", file = TRUE)
    )
  }
}

arch_scales <- list(
  scale_colour_manual(values = pal, breaks = arch_order, labels = arch_labels[arch_order]),
  scale_fill_manual(values = pal, breaks = arch_order, labels = arch_labels[arch_order])
)

# -------------------------------------------------------------------------
# Figure 1: study logic and theoretical design
# -------------------------------------------------------------------------
wave <- tibble(x = seq(0.07, 0.20, length.out = 120)) %>%
  mutate(y = 0.70 + 0.075 * sin((x - 0.07) * 125) * exp(-((x - 0.135) / 0.06)^2))
traj <- tibble(x = c(0.34, 0.45, 0.55, 0.65), y = c(0.56, 0.64, 0.58, 0.74))
future <- tibble(x = 0.89, y = c(0.52, 0.69, 0.86))
metrics <- tibble(x = c(0.18, 0.60, 0.18, 0.60), y = c(0.16, 0.16, 0.07, 0.07),
                  label = c("Gain", "Broadcast", "Persistence", "Concentration"))
p1a <- ggplot() +
  annotate("rect", xmin = 0.05, xmax = 0.22, ymin = 0.58, ymax = 0.82,
           fill = NA, colour = red, size = 0.7) +
  geom_path(data = wave, aes(x, y), colour = red, size = 0.8) +
  annotate("segment", x = 0.23, xend = 0.32, y = 0.70, yend = 0.70,
           colour = red, size = 0.65, arrow = grid::arrow(length = grid::unit(2.2, "mm"))) +
  geom_path(data = traj, aes(x, y), colour = ink, size = 1.0) +
  geom_point(data = traj, aes(x, y), shape = 21, size = 3.2, fill = paper, colour = mid, stroke = 0.7) +
  geom_curve(data = future, aes(x = 0.66, y = 0.73, xend = x - 0.025, yend = y),
             curvature = 0.10, colour = red, size = 0.55,
             arrow = grid::arrow(length = grid::unit(1.9, "mm"))) +
  geom_point(data = future, aes(x, y), shape = 21, size = 3.0, fill = paper, colour = mid, stroke = 0.65) +
  annotate("text", x = c(0.135, 0.49, 0.87), y = c(0.51, 0.49, 0.42),
           label = c("stimulus", "state trajectory", "future computation"), size = 2.35, colour = ink) +
  annotate("text", x = 0.50, y = 0.31,
           label = "J(t) = d future outputs / d state(t)", size = 2.6, colour = ink) +
  geom_point(data = metrics, aes(x, y), shape = 21, size = 2.8, fill = paper, colour = mid, stroke = 0.6) +
  geom_text(data = metrics, aes(x + 0.045, y, label = label), hjust = 0, size = 2.05, colour = ink) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "    A common causal geometry") + theme_void_sa()

centers <- seq(0.10, 0.90, length.out = 5)
mechanism_labels <- c("Feedforward", "Recurrent", "Private\nmodules", "Limited\nshared", "Unlimited\nshared")
nodes <- bind_rows(
  tibble(architecture = arch_order[1], x = centers[1] + c(-0.06, 0, 0.06), y = 0.68, size = 3),
  tibble(architecture = arch_order[2], x = centers[2] + c(-0.05, 0.05), y = 0.68, size = 3),
  tibble(architecture = arch_order[3], x = centers[3] + rep(c(-0.05, 0.05), each = 2), y = rep(c(0.73, 0.62), 2), size = 2.7),
  tibble(architecture = arch_order[4], x = centers[4] + c(-0.065, -0.065, 0, 0.065, 0.065), y = c(0.74, 0.61, 0.675, 0.74, 0.61), size = c(2.6, 2.6, 5.5, 2.6, 2.6)),
  tibble(architecture = arch_order[5], x = centers[5] + c(-0.065, -0.065, 0, 0.065, 0.065), y = c(0.74, 0.61, 0.675, 0.74, 0.61), size = c(2.6, 2.6, 9, 2.6, 2.6))
)
edges <- bind_rows(
  tibble(x = centers[1] + c(-0.05, 0.01), xend = centers[1] + c(-0.01, 0.05), y = 0.68, yend = 0.68),
  tibble(x = centers[2] - 0.04, xend = centers[2] + 0.04, y = 0.68, yend = 0.68),
  tibble(x = centers[3] + c(-0.05, 0.05), xend = centers[3] + c(-0.05, 0.05), y = 0.71, yend = 0.64),
  tibble(x = rep(centers[4:5], each = 4) + rep(c(-0.055, -0.055, 0.055, 0.055), 2),
         xend = rep(centers[4:5], each = 4),
         y = rep(c(0.73, 0.62, 0.73, 0.62), 2), yend = 0.675)
)
p1b <- ggplot() +
  geom_segment(data = edges, aes(x, y, xend = xend, yend = yend), colour = mid, size = 0.45) +
  geom_point(data = nodes, aes(x, y, fill = architecture, size = size), shape = 21, colour = ink, stroke = 0.55) +
  scale_size_identity() + scale_fill_manual(values = pal) +
  annotate("text", x = centers, y = 0.47, label = mechanism_labels, size = 2.1, lineheight = 0.9) +
  annotate("text", x = 0.50, y = 0.24,
           label = "Does sharing require a capacity bottleneck\nto match human geometry?",
           colour = red, fontface = "bold", size = 2.75, lineheight = 0.96) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "    Five competing mechanisms") + theme_void_sa() + theme(legend.position = "none")

flow <- tibble(x = seq(0.10, 0.90, length.out = 5), y = 0.62,
               icon = c("EEG", "J", "5M", "ADAPT", "TEST"),
               label = c("EEG", "Geometry", "Models", "Adapt", "Causal test"))
p1c <- ggplot(flow, aes(x, y)) +
  geom_segment(data = tibble(x = flow$x[-5] + 0.065, xend = flow$x[-1] - 0.065, y = 0.62, yend = 0.62),
               aes(x, y, xend = xend, yend = yend), inherit.aes = FALSE,
               colour = mid, size = 0.55, arrow = grid::arrow(length = grid::unit(1.8, "mm"))) +
  geom_point(shape = 21, size = 10.5, fill = "white", colour = mid, stroke = 0.9) +
  geom_text(aes(label = icon), size = 2.45, fontface = "bold", colour = red) +
  geom_text(aes(y = 0.39, label = label), size = 2.05, colour = ink) +
  annotate("text", x = 0.5, y = 0.16,
           label = "Independent construction -> held-out causal evaluation",
           size = 2.1, colour = ink) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "    One inferential chain") + theme_void_sa()

safeguards <- tibble(
  x = rep(c(0.08, 0.56), 3), y = rep(c(0.80, 0.52, 0.24), each = 2),
  head = c("Discovery", "Replication", "Held out", "Sham", "Causal", "Retention"),
  body = c("Gabor access contrast", "Kronemer + Somato", "Participants + seeds",
           "Joint time-label permutation", "Targeted vs random subspaces", "Task change <= 0.02 + audit"),
  emphasis = c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE)
)
p1d <- ggplot(safeguards, aes(x, y)) +
  geom_point(aes(fill = emphasis), shape = 21, size = 4.1, colour = mid, stroke = 0.55) +
  scale_fill_manual(values = c(`TRUE` = red, `FALSE` = paper), guide = "none") +
  geom_text(aes(x = x + 0.04, y = y + 0.025, label = head), hjust = 0, fontface = "bold", size = 2.4, colour = ink) +
  geom_text(aes(x = x + 0.04, y = y - 0.035, label = body), hjust = 0, size = 1.95, colour = mid) +
  annotate("text", x = 0.5, y = 0.05, label = "No architecture receives a composite score.",
           fontface = "bold", size = 2.2, colour = ink) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "    Safeguards and outcome-neutral interpretation") + theme_void_sa()

fig1 <- ((p1a + p1b) + plot_layout(widths = c(0.44, 0.56))) /
  ((p1c + p1d) + plot_layout(widths = c(0.44, 0.56))) +
  plot_annotation(tag_levels = "a")
save_figure(fig1, "Fig1-study-logic-R", "Question, theory space, and study logic", height = 5.15)

# -------------------------------------------------------------------------
# Figure 2: human geometry
# -------------------------------------------------------------------------
timecourses <- read_fig("human-timecourses.csv") %>%
  filter(is.finite(mean), is.finite(lower), is.finite(upper))
timecourses$dataset <- factor(timecourses$dataset, levels = names(dataset_pal),
                              labels = c("Gabor discovery", "Kronemer replication", "Somato replication"))
p2a <- ggplot(timecourses, aes(time_ms, mean, group = contrast_id,
                               colour = dataset, fill = dataset, linetype = contrast_id)) +
  annotate("rect", xmin = 150, xmax = 300, ymin = -Inf, ymax = Inf, fill = paper, alpha = 0.32) +
  geom_hline(yintercept = 0, colour = ink, size = 0.3) +
  geom_vline(xintercept = 0, colour = ink, size = 0.3) +
  geom_ribbon(aes(ymin = lower, ymax = upper), colour = NA, alpha = 0.12) +
  geom_line(size = 0.7) +
  facet_wrap(~dataset, nrow = 1, scales = "free_y") +
  scale_colour_manual(values = unname(dataset_pal)) +
  scale_fill_manual(values = unname(dataset_pal)) +
  scale_linetype_manual(values = c("solid", "22", "42", "13", "73")) +
  labs(x = "Time from stimulus (ms)", y = "Access Index difference") +
  theme_sa() + theme(legend.position = "none")

effects <- read_fig("human-participant-effects.csv") %>%
  mutate(dataset = factor(dataset, levels = names(dataset_pal)),
         contrast_id = factor(contrast_id, levels = unique(contrast_id)))
effect_summary <- effects %>% group_by(contrast_id, dataset) %>%
  summarise(mean_ci(contrast), .groups = "drop")
p2b <- ggplot(effects, aes(contrast, contrast_id, colour = dataset)) +
  geom_jitter(height = 0.10, width = 0, alpha = 0.22, size = 0.9) +
  geom_segment(data = effect_summary, aes(x = lower, xend = upper, y = contrast_id, yend = contrast_id),
               size = 0.65) +
  geom_point(data = effect_summary, aes(x = mean, y = contrast_id), shape = 21, fill = "white", size = 2.6, stroke = 0.8) +
  geom_vline(xintercept = 0, colour = ink, size = 0.35) +
  scale_colour_manual(values = dataset_pal) +
  labs(x = "Primary-window contrast (95% CI)", y = NULL, title = "Participant-level effects") +
  theme_sa() + theme(legend.position = "none")

human_fp <- read_fig("human-fingerprints.csv") %>%
  mutate(metric = as_metric(metric), contrast_id = factor(contrast_id, levels = unique(contrast_id)))
fp_limit <- max(abs(human_fp$standardized_contrast), na.rm = TRUE)
p2c <- ggplot(human_fp, aes(metric, contrast_id, fill = standardized_contrast)) +
  geom_tile(colour = "white", size = 0.45) + div_scale(max(1, fp_limit), "Discovery-SD units") +
  labs(x = NULL, y = NULL, title = "Human geometry fingerprints") +
  theme_sa() + theme(axis.text.x = element_text(angle = 28, hjust = 1),
                     legend.position = "bottom", legend.key.width = grid::unit(9, "mm"))

pred <- read_fig("prediction-folds.csv") %>%
  mutate(family = factor(family, levels = c("conventional", "conventional_plus_jacobian"),
                         labels = c("Conventional", "+ Jacobian")))
delta_auc <- pred %>% select(fold, family, auc) %>% pivot_wider(names_from = family, values_from = auc) %>%
  summarise(delta = mean(`+ Jacobian` - Conventional)) %>% pull(delta)
p2d <- ggplot(pred, aes(family, auc, group = fold)) +
  geom_line(colour = sand, size = 0.55) +
  geom_point(aes(fill = family), shape = 21, size = 2.1, colour = ink, stroke = 0.45) +
  scale_fill_manual(values = c("Conventional" = mid, "+ Jacobian" = red), guide = "none") +
  annotate("text", x = 1.95, y = max(pred$auc), hjust = 1, vjust = 1.25,
           label = sprintf("mean delta AUC = %.4f", delta_auc), size = 2.1, colour = ink) +
  labs(x = NULL, y = "Held-out AUC", title = "Incremental prediction") + theme_sa()

fig2 <- p2a / (p2b + p2c + p2d) + plot_layout(heights = c(1.05, 1)) +
  plot_annotation(tag_levels = "a")
save_figure(fig2, "Fig2-human-geometry-R", "Human access geometry is heterogeneous and adds no prediction")

# -------------------------------------------------------------------------
# Figure 3: machine geometry and causal controls
# -------------------------------------------------------------------------
accuracy <- read_fig("machine-accuracy.csv") %>% mutate(architecture_raw = architecture, architecture = as_arch(architecture))
selected_bin <- accuracy %>% filter(selected_common_bin) %>% distinct(difficulty_bin) %>% pull(difficulty_bin)
p3a <- ggplot(accuracy, aes(difficulty_bin, mean_accuracy, colour = architecture_raw)) +
  geom_line(size = 0.75) + geom_point(size = 1.55) +
  geom_vline(xintercept = selected_bin[1], linetype = "dashed", colour = ink, size = 0.4) +
  scale_colour_manual(values = pal, breaks = arch_order, labels = arch_labels[arch_order], name = NULL) +
  labs(x = "Difficulty bin", y = "Presence accuracy", title = "Accuracy-matched comparison") +
  theme_sa() + theme(legend.position = "bottom", legend.text = element_text(size = 5.8)) +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE))

steps <- read_fig("machine-step-gain-seed.csv") %>%
  group_by(architecture, step) %>% summarise(mean_ci(gain), .groups = "drop")
p3b <- ggplot(steps, aes(step, mean, colour = architecture, fill = architecture)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), colour = NA, alpha = 0.12) +
  geom_line(size = 0.75) + geom_point(size = 1.6) +
  scale_colour_manual(values = pal) + scale_fill_manual(values = pal) +
  labs(x = "Processing step", y = "Mean Jacobian gain", title = "Step-resolved dynamics") +
  theme_sa() + theme(legend.position = "none")

machine_int <- read_fig("machine-interventions.csv") %>%
  select(architecture, seed, random_drop_mean, top_subspace_accuracy_drop) %>%
  pivot_longer(c(random_drop_mean, top_subspace_accuracy_drop), names_to = "condition", values_to = "drop") %>%
  mutate(arch_index = match(architecture, arch_order),
         offset = if_else(condition == "random_drop_mean", -0.12, 0.12), x = arch_index + offset)
p3c <- ggplot(machine_int, aes(x, drop, group = interaction(architecture, seed))) +
  geom_line(colour = alpha(sand, 0.65), size = 0.35) +
  geom_point(data = filter(machine_int, condition == "random_drop_mean"),
             aes(colour = architecture), shape = 21, fill = "white", size = 1.55, stroke = 0.55) +
  geom_point(data = filter(machine_int, condition == "top_subspace_accuracy_drop"),
             aes(fill = architecture, colour = architecture), shape = 21, size = 1.55, stroke = 0.55) +
  scale_x_continuous(breaks = 1:5, labels = arch_short[arch_order]) +
  scale_colour_manual(values = outline) + scale_fill_manual(values = pal) +
  labs(x = NULL, y = "Presence-accuracy drop", title = "Targeted versus random intervention",
       subtitle = "open = random; filled = top-4 subspace") +
  theme_sa() + theme(axis.text.x = element_text(angle = 24, hjust = 1), legend.position = "none")

machine_fp <- read_fig("machine-fingerprints.csv") %>% mutate(architecture = as_arch(architecture), metric = as_metric(metric))
machine_limit <- max(abs(machine_fp$contrast), na.rm = TRUE)
p3d <- ggplot(machine_fp, aes(metric, architecture, fill = contrast)) +
  geom_tile(colour = "white", size = 0.5) + div_scale(machine_limit, "Correct - incorrect (z)") +
  labs(x = NULL, y = NULL, title = "Accuracy-matched machine fingerprints") +
  theme_sa() + theme(axis.text.x = element_text(angle = 28, hjust = 1),
                     legend.position = "bottom", legend.key.width = grid::unit(10, "mm"))

fig3 <- (p3a + p3b) / (p3c + p3d) + plot_annotation(tag_levels = "a")
save_figure(fig3, "Fig3-machine-geometry-R", "Architecture controls and causal machine geometry")

# -------------------------------------------------------------------------
# Figure 4: human-machine convergence
# -------------------------------------------------------------------------
stages <- read_fig("seed-stage-summary.csv")
baseline <- stages %>% filter(stage == "task_trained") %>% mutate(architecture_factor = as_arch(architecture))
baseline_summary <- baseline %>% group_by(architecture, architecture_factor) %>% summarise(mean_ci(rms_distance), .groups = "drop")
set.seed(20260819)
p4a <- ggplot(baseline, aes(architecture_factor, rms_distance, colour = architecture)) +
  geom_jitter(width = 0.10, alpha = 0.34, size = 1.25) +
  geom_errorbar(data = baseline_summary, aes(y = mean, ymin = lower, ymax = upper), width = 0.12, colour = ink, size = 0.55) +
  geom_point(data = baseline_summary, aes(y = mean), shape = 23, fill = ink, colour = ink, size = 2.3) +
  scale_colour_manual(values = pal) +
  labs(x = NULL, y = "Held-out human RMS distance\n(lower is closer)", title = "Baseline human-machine convergence") +
  theme_sa() + theme(axis.text.x = element_text(angle = 24, hjust = 1), legend.position = "none")

profiles <- read_fig("geometry-profiles.csv") %>%
  mutate(metric = as_metric(metric), architecture = if_else(profile == "human_discovery", "human_discovery", profile))
profile_cols <- c(human_discovery = red, pal)
profile_labs <- c(human_discovery = "Human discovery", arch_labels)
p4b <- ggplot(profiles, aes(metric, value, group = profile, colour = profile)) +
  geom_hline(yintercept = 0, colour = ink, size = 0.3) +
  geom_line(aes(size = profile_type)) + geom_point(aes(size = profile_type)) +
  scale_colour_manual(values = profile_cols, labels = profile_labs[names(profile_cols)], name = NULL) +
  scale_size_manual(values = c(human = 1.15, machine = 0.62), guide = "none") +
  labs(x = NULL, y = "Standardized contrast", title = "Geometry profiles") +
  theme_sa() + theme(axis.text.x = element_text(angle = 20, hjust = 1), legend.position = "bottom",
                     legend.text = element_text(size = 5.5)) + guides(colour = guide_legend(nrow = 2))

equiv <- read_fig("capacity-equivalence.csv") %>% mutate(metric = as_metric(metric))
p4c <- ggplot(equiv, aes(mean, metric)) +
  annotate("rect", xmin = -0.20, xmax = 0.20, ymin = -Inf, ymax = Inf, fill = paper, alpha = 0.62) +
  geom_vline(xintercept = 0, colour = ink, size = 0.35) +
  geom_segment(aes(x = lower, xend = upper, yend = metric), colour = mid, size = 0.7) +
  geom_point(shape = 21, fill = mid, colour = mid, size = 2.5) +
  coord_cartesian(xlim = c(-0.25, 0.35)) +
  annotate("text", x = -0.235, y = 4.25, label = "shaded: +/-0.20 equivalence margin", hjust = 0, size = 2.0, colour = ink) +
  labs(x = "Limited - unlimited (90% CI)", y = NULL, title = "Capacity-limit equivalence test") + theme_sa()

lodo <- read_fig("lodo-generalization.csv") %>% mutate(dataset = held_out_dataset)
p4d <- ggplot(lodo, aes(standardized_distance, cosine_similarity, fill = dataset, label = contrast_id)) +
  geom_point(shape = 21, size = 3.0, colour = ink, stroke = 0.55) +
  geom_text_repel(size = 2.2, family = "Liberation Sans", colour = ink,
                  min.segment.length = 0, segment.size = 0.3, box.padding = 0.22) +
  scale_fill_manual(values = dataset_pal) +
  labs(x = "LODO standardized distance\n(lower is closer)", y = "LODO cosine similarity",
       title = "Discovery-to-replication generalization") + theme_sa() + theme(legend.position = "none")

fig4 <- (p4a + p4b) / (p4c + p4d) + plot_annotation(tag_levels = "a")
save_figure(fig4, "Fig4-human-machine-convergence-R", "Human-machine convergence adjudicates theories")

# -------------------------------------------------------------------------
# Figure 5: human-guided adaptation and retention audit
# -------------------------------------------------------------------------
adapt_flow <- tibble(x = seq(0.09, 0.91, length.out = 5), y = 0.66,
                     icon = c("EEG", "LOCK", "ADAPT", "HOLD", "GATE"),
                     label = c("EEG RSM\n150-300 ms", "Frozen encoder\nand heads", "Adapt internal\ndynamics",
                               "Held-out\nparticipants", "Geometry +\ntask gate"))
p5a <- ggplot(adapt_flow, aes(x, y)) +
  geom_segment(data = tibble(x = adapt_flow$x[-5] + 0.06, xend = adapt_flow$x[-1] - 0.06, y = 0.66, yend = 0.66),
               aes(x, y, xend = xend, yend = yend), inherit.aes = FALSE, colour = mid, size = 0.55,
               arrow = grid::arrow(length = grid::unit(1.8, "mm"))) +
  geom_point(shape = 21, size = 10, fill = "white", colour = mid, stroke = 0.85) +
  geom_text(aes(label = icon), size = 2.6, fontface = "bold", colour = ink) +
  geom_text(aes(y = 0.44, label = label), size = 1.9, lineheight = 0.92) +
  annotate("segment", x = 0.50, xend = 0.36, y = 0.58, yend = 0.24,
           colour = red, size = 0.55, arrow = grid::arrow(length = grid::unit(1.7, "mm"))) +
  annotate("segment", x = 0.52, xend = 0.69, y = 0.58, yend = 0.24,
           colour = mid, size = 0.55, arrow = grid::arrow(length = grid::unit(1.7, "mm"))) +
  annotate("text", x = c(0.34, 0.71), y = 0.16, label = c("human target", "time-permuted sham"),
           colour = c(red, mid), size = 2.25) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(title = "    Held-out adaptation design") + theme_void_sa()

stage_summary <- stages %>% filter(stage %in% stage_order) %>%
  group_by(architecture, stage) %>% summarise(mean_ci(rms_distance), .groups = "drop") %>%
  mutate(stage = factor(stage, levels = stage_order, labels = stage_labels[stage_order]))
p5b <- ggplot(stage_summary, aes(stage, mean, group = architecture, colour = architecture, fill = architecture)) +
  geom_line(size = 0.7) + geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.08, size = 0.5) +
  geom_point(shape = 21, size = 2.0, stroke = 0.5) +
  scale_colour_manual(values = outline, breaks = arch_order, labels = arch_labels[arch_order], name = NULL) +
  scale_fill_manual(values = pal, breaks = arch_order, labels = arch_labels[arch_order], name = NULL) +
  labs(x = NULL, y = "Held-out RMS distance", title = "State-geometry trajectories") +
  theme_sa() + theme(axis.text.x = element_text(angle = 22, hjust = 1), legend.position = "bottom",
                     legend.text = element_text(size = 5.6)) + guides(colour = guide_legend(nrow = 2))

sham <- read_fig("sham-comparison.csv")
sham_seed <- sham %>% group_by(architecture, seed) %>% summarise(human_specific_gain = mean(human_specific_gain), .groups = "drop") %>%
  mutate(architecture_factor = as_arch(architecture), arch_index = match(architecture, arch_order))
sham_all <- sham_seed %>% group_by(architecture, architecture_factor) %>% summarise(mean_ci(human_specific_gain), .groups = "drop") %>% mutate(type = "All runs")
complete <- read_fig("retention-complete-case.csv") %>% mutate(architecture_factor = as_arch(architecture), type = "Complete case")
summary_sham <- bind_rows(
  transmute(sham_all, architecture, architecture_factor, type, mean, lower, upper),
  transmute(complete, architecture, architecture_factor, type, mean, lower, upper)
) %>% mutate(offset = if_else(type == "All runs", -0.08, 0.08), x = match(architecture, arch_order) + offset)
set.seed(20260819)
p5c <- ggplot() +
  geom_hline(yintercept = 0, colour = ink, size = 0.35) +
  geom_jitter(data = sham_seed, aes(arch_index, human_specific_gain, fill = architecture),
              shape = 21, colour = alpha(ink, 0.55), stroke = 0.3,
              width = 0.10, alpha = 0.30, size = 1.15) +
  geom_errorbar(data = summary_sham, aes(x = x, ymin = lower, ymax = upper, colour = type), width = 0.05, size = 0.55) +
  geom_point(data = summary_sham, aes(x = x, y = mean, colour = type, shape = type), fill = "white", size = 2.3, stroke = 0.7) +
  scale_x_continuous(breaks = 1:5, labels = arch_short[arch_order]) +
  scale_fill_manual(values = pal, guide = "none") +
  scale_colour_manual(values = c(`All runs` = ink, `Complete case` = red), name = NULL) +
  scale_shape_manual(values = c(`All runs` = 23, `Complete case` = 22), name = NULL) +
  labs(x = NULL, y = "Sham distance - human-adapted distance", title = "Human-specific gain over sham") +
  theme_sa() + theme(axis.text.x = element_text(angle = 24, hjust = 1), legend.position = "top")

costs <- read_fig("adaptation-cost.csv") %>% filter(condition == "human_adapted") %>%
  mutate(gate = if_else(performance_gate_passed, "Inside gate", "Outside gate"))
p5d <- ggplot(costs, aes(relative_l2_parameter_displacement, accuracy_change)) +
  geom_hline(yintercept = c(-0.02, 0.02), linetype = "dashed", colour = mid, size = 0.45) +
  geom_hline(yintercept = 0, colour = ink, size = 0.32) +
  geom_point(data = filter(costs, gate == "Inside gate"), aes(fill = architecture), shape = 21,
             colour = alpha(ink, 0.55), size = 1.35, alpha = 0.45, stroke = 0.35) +
  geom_point(data = filter(costs, gate == "Outside gate"), colour = red, shape = 4, size = 2.0, stroke = 0.7) +
  scale_fill_manual(values = pal) +
  annotate("text", x = Inf, y = 0.0165, label = "x: 17/500 human-adapted runs outside gate",
           hjust = 1.03, vjust = 0, size = 2.05, colour = red) +
  labs(x = "Relative L2 parameter displacement", y = "Task-accuracy change",
       title = "Retention gate and adaptation cost") + theme_sa() + theme(legend.position = "none")

fig5 <- (p5a + p5b) / (p5c + p5d) + plot_annotation(tag_levels = "a")
save_figure(fig5, "Fig5-human-guided-adaptation-R", "Human-guided adaptation does not improve held-out geometry")

# -------------------------------------------------------------------------
# Figure 6: causal transfer and integrated evidence
# -------------------------------------------------------------------------
interventions <- read_fig("post-adaptation-interventions.csv")
causal_summary <- interventions %>% filter(stage %in% c("task_trained", "human_adapted")) %>%
  group_by(architecture, stage, seed) %>% summarise(causal_specificity = mean(causal_specificity), .groups = "drop") %>%
  group_by(architecture, stage) %>% summarise(mean_ci(causal_specificity), .groups = "drop") %>%
  mutate(x = match(architecture, arch_order) + if_else(stage == "task_trained", -0.11, 0.11),
         stage_label = if_else(stage == "task_trained", "Task trained", "Human adapted"))
p6a <- ggplot(causal_summary, aes(x, mean, colour = architecture, fill = architecture, shape = stage_label)) +
  geom_hline(yintercept = 0, colour = ink, size = 0.32) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.06, size = 0.55) +
  geom_point(size = 2.5, stroke = 0.65) +
  scale_x_continuous(breaks = 1:5, labels = arch_short[arch_order]) +
  scale_colour_manual(values = outline, guide = "none") +
  scale_fill_manual(values = pal, guide = "none") +
  scale_shape_manual(values = c(`Task trained` = 21, `Human adapted` = 23), name = NULL) +
  labs(x = NULL, y = "Targeted drop - random drop", title = "Causal specificity") +
  theme_sa() + theme(axis.text.x = element_text(angle = 24, hjust = 1), legend.position = "top")

transfer <- read_fig("transfer-tests.csv") %>%
  filter(evaluation_contrast %in% c("kronemer-0", "kronemer-1", "somato-0", "somato-1")) %>%
  mutate(architecture_factor = as_arch(architecture), evaluation_contrast = factor(evaluation_contrast,
         levels = c("kronemer-0", "kronemer-1", "somato-0", "somato-1")))
transfer_limit <- max(abs(transfer$mean_alignment_gain), na.rm = TRUE)
p6b <- ggplot(transfer, aes(evaluation_contrast, architecture_factor, fill = mean_alignment_gain)) +
  geom_tile(colour = "white", size = 0.45) +
  geom_text(aes(label = sprintf("%+.02f", mean_alignment_gain),
                colour = abs(mean_alignment_gain) > 0.65 * transfer_limit), size = 2.15) +
  scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = ink), guide = "none") +
  scale_y_discrete(limits = rev(levels(transfer$architecture_factor))) +
  div_scale(transfer_limit, "Alignment gain") +
  labs(x = NULL, y = NULL, title = "External transfer after Gabor adaptation") +
  theme_sa() + theme(axis.text.x = element_text(angle = 25, hjust = 1),
                     legend.position = "right", legend.key.height = grid::unit(12, "mm"))

endpoints <- read_fig("architecture-endpoints.csv")
primary <- read_fig("primary-tests.csv") %>% select(architecture, primary_mean = mean)
sham_tests <- read_fig("sham-tests.csv") %>% select(architecture, sham_mean = mean)
causal_tests <- read_fig("intervention-tests.csv") %>% select(architecture, causal_mean = mean_causal_specificity_change)
pass_rates <- costs %>% group_by(architecture) %>% summarise(pass_rate = mean(performance_gate_passed), .groups = "drop")
transfer_pos <- transfer %>%
  mutate(ci_lower = as.numeric(sub("^\\[([^,]+),.*$", "\\1", confidence_interval_95))) %>%
  group_by(architecture) %>% summarise(positive = sum(ci_lower > 0), .groups = "drop")
evidence <- endpoints %>% left_join(primary, by = "architecture") %>% left_join(sham_tests, by = "architecture") %>%
  left_join(causal_tests, by = "architecture") %>% left_join(pass_rates, by = "architecture") %>%
  left_join(transfer_pos, by = "architecture") %>%
  mutate(rank = rank(heldout_rms_distance, ties.method = "first"), architecture_factor = as_arch(architecture))
grid_long <- bind_rows(
  transmute(evidence, architecture, architecture_factor, column = "Baseline\nfit", value = sprintf("#%d\n%.2f", rank, heldout_rms_distance)),
  transmute(evidence, architecture, architecture_factor, column = "Human\nadapt.", value = sprintf("%+.3f", primary_mean)),
  transmute(evidence, architecture, architecture_factor, column = "Sham\nspecific.", value = sprintf("%+.4f", sham_mean)),
  transmute(evidence, architecture, architecture_factor, column = "External\ntransfer", value = sprintf("%d/4 +", positive)),
  transmute(evidence, architecture, architecture_factor, column = "Causal\nchange", value = sprintf("%+.3f", causal_mean)),
  transmute(evidence, architecture, architecture_factor, column = "Task\nretention", value = sprintf("%.0f%%", 100 * pass_rate))
) %>% mutate(column = factor(column, levels = c("Baseline\nfit", "Human\nadapt.", "Sham\nspecific.",
                                               "External\ntransfer", "Causal\nchange", "Task\nretention")),
             face = if_else(column == "Baseline\nfit", "bold", "plain"))
architecture_dots <- distinct(grid_long, architecture, architecture_factor) %>%
  mutate(column = factor("Baseline\nfit", levels = levels(grid_long$column)))
p6c <- ggplot(grid_long, aes(column, architecture_factor)) +
  geom_tile(fill = "white", colour = paper, size = 0.55) +
  geom_text(aes(label = value, fontface = face), size = 2.15, lineheight = 0.9, colour = ink) +
  geom_point(data = architecture_dots, aes(column, architecture_factor, fill = architecture),
             inherit.aes = FALSE, position = position_nudge(x = -0.34),
             shape = 21, size = 2.2, colour = mid, stroke = 0.45) +
  scale_fill_manual(values = pal) +
  scale_x_discrete(position = "top") +
  scale_y_discrete(limits = rev(levels(grid_long$architecture_factor))) +
  labs(x = NULL, y = NULL, title = "Prespecified evidence map", subtitle = "No composite score") +
  coord_cartesian(clip = "off") + theme_sa() +
  theme(panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank(),
        axis.text.x = element_text(size = 6.2), legend.position = "none",
        plot.margin = margin(12, 5, 5, 8))

gain_seed <- sham %>% group_by(architecture, seed) %>% summarise(alignment_gain = mean(alignment_gain), .groups = "drop")
task_seed <- interventions %>% filter(stage == "task_trained") %>% select(architecture, seed, task = causal_specificity)
adapt_seed <- interventions %>% filter(stage == "human_adapted") %>% group_by(architecture, seed) %>%
  summarise(adapted = mean(causal_specificity), .groups = "drop")
trade <- gain_seed %>% inner_join(task_seed, by = c("architecture", "seed")) %>%
  inner_join(adapt_seed, by = c("architecture", "seed")) %>% mutate(causal_change = adapted - task)
trade_mean <- trade %>% group_by(architecture) %>% summarise(alignment_gain = mean(alignment_gain),
                                                              causal_change = mean(causal_change), .groups = "drop")
pareto <- trade_mean %>% rowwise() %>% mutate(dominated = any(
  trade_mean$alignment_gain >= alignment_gain & trade_mean$causal_change >= causal_change &
    (trade_mean$alignment_gain > alignment_gain | trade_mean$causal_change > causal_change)
)) %>% ungroup() %>% filter(!dominated) %>% arrange(alignment_gain)
p6d <- ggplot(trade, aes(alignment_gain, causal_change, colour = architecture, fill = architecture)) +
  geom_hline(yintercept = 0, colour = ink, size = 0.32) + geom_vline(xintercept = 0, colour = ink, size = 0.32) +
  geom_point(shape = 21, size = 1.35, alpha = 0.28, stroke = 0.35) +
  geom_path(data = pareto, colour = red, linetype = "dashed", size = 0.55, inherit.aes = FALSE,
            aes(alignment_gain, causal_change)) +
  geom_point(data = trade_mean, shape = 23, size = 3.0, stroke = 0.65) +
  geom_text_repel(data = trade_mean, aes(label = arch_short[architecture]), size = 2.15,
                  family = "Liberation Sans", colour = ink, min.segment.length = 0,
                  segment.size = 0.3, box.padding = 0.25, point.padding = 0.2) +
  scale_colour_manual(values = outline) + scale_fill_manual(values = pal) +
  labs(x = "Held-out alignment gain", y = "Causal-specificity change",
       title = "Alignment-causality trade-off") + theme_sa() + theme(legend.position = "none")

fig6 <- (p6a + p6b) / (p6c + p6d) + plot_annotation(tag_levels = "a")
save_figure(fig6, "Fig6-integrated-evidence-R", "Causal transfer and integrated theoretical evidence", height = 5.85)

manifest <- list(
  ready = TRUE,
  design = "integrated_science_advances_six_figure_system_native_r",
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  renderer = list(language = "R", version = R.version.string,
                  packages = list(ggplot2 = as.character(packageVersion("ggplot2")),
                                  patchwork = as.character(packageVersion("patchwork")),
                                  svglite = as.character(packageVersion("svglite")))),
  palette = unname(pal),
  main_figures = 6,
  files = files_out,
  data_manifest = normalizePath(file.path(data_dir, "manifest.json"), winslash = "/", mustWork = TRUE)
)
dir.create(dirname(manifest_path), recursive = TRUE, showWarnings = FALSE)
write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
cat(toJSON(list(ready = TRUE, files = length(files_out), output = output_dir), auto_unbox = TRUE, pretty = TRUE), "\n")
