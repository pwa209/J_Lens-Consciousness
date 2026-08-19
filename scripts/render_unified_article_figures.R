#!/usr/bin/env Rscript

# Unified AAAS-oriented figure system for one study with linked analytic stages.
# Base R only: vector PDF and 300-dpi PNG are produced from the same source.

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default = NULL) {
  hit <- match(flag, args)
  if (is.na(hit) || hit == length(args)) return(default)
  args[[hit + 1L]]
}

mock_mode <- "--mock" %in% args
output_root <- arg_value("--output", "output/pdf/unified-article-figures")
baseline_root <- arg_value("--baseline-root", "publication-results")
adaptation_root <- arg_value("--adaptation-root", "results-extension/human-adaptation")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  feedforward = "#184D77",
  private_modules = "#497987",
  recurrent = "#858D7E",
  constrained_shared = "#D48448",
  unlimited_shared = "#BE4A36",
  control = "#E1BD89"
)
ink <- "#17232C"
mid_ink <- "#59636A"
light_ink <- "#A7AFB2"
paper <- "#FFFFFF"
panel_fill <- "#FFFFFF"
white <- "#FFFFFF"
arch <- names(pal)[1:5]
arch_label <- c("Feedforward", "Private modules", "Recurrent",
                "Constrained shared", "Unlimited shared")
arch_code <- LETTERS[1:5]

alpha <- function(colour, opacity) adjustcolor(colour, alpha.f = opacity)
se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(is.finite(x)))
ci <- function(x) c(mean(x, na.rm = TRUE), 1.96 * se(x))

set_theme <- function() {
  par(family = "sans", fg = ink, col.axis = ink, col.lab = ink,
      las = 1, lend = "round", ljoin = "round", xaxs = "i", yaxs = "i")
}

canvas <- function() {
  plot.new()
  plot.window(c(0, 1), c(0, 1), xaxs = "i", yaxs = "i")
}

figure_title <- function(title, subtitle = NULL) {
  mtext(title, side = 3, outer = TRUE, line = 1.15, adj = 0,
        cex = 1.10, font = 2, col = ink)
  if (!is.null(subtitle)) {
    mtext(subtitle, side = 3, outer = TRUE, line = .18, adj = 0,
          cex = .67, col = mid_ink)
  }
}

panel_label <- function(label, title = NULL, x = .01, y = 1.06) {
  usr <- par("usr")
  xx <- usr[1] + x * (usr[2] - usr[1])
  yy <- usr[3] + y * (usr[4] - usr[3])
  tx <- usr[1] + (x + .065) * (usr[2] - usr[1])
  ty <- usr[3] + (y - .005) * (usr[4] - usr[3])
  text(xx, yy, label, adj = c(0, 1), font = 2, cex = 1.35, col = ink, xpd = NA)
  if (!is.null(title)) text(tx, ty, title, adj = c(0, 1),
                            font = 2, cex = .88, col = ink, xpd = NA)
}

mock_stamp <- function() {
  text(.995, .03, "DESIGN MOCK - ILLUSTRATIVE VALUES", srt = 90,
       adj = c(0, 1), cex = .60, col = alpha(mid_ink, .65), xpd = NA)
}

round_box <- function(x0, y0, x1, y1, fill = panel_fill, border = NA,
                      radius = .012, lwd = 1.2) {
  rect(x0 + radius, y0, x1 - radius, y1, col = fill, border = NA)
  rect(x0, y0 + radius, x1, y1 - radius, col = fill, border = NA)
  symbols(c(x0 + radius, x1 - radius, x0 + radius, x1 - radius),
          c(y0 + radius, y0 + radius, y1 - radius, y1 - radius),
          circles = rep(radius, 4), inches = FALSE, add = TRUE,
          bg = fill, fg = NA)
  if (!is.na(border)) rect(x0, y0, x1, y1, border = border, lwd = lwd)
}

arrow_link <- function(x0, y0, x1, y1, col = mid_ink, lwd = 1.5) {
  arrows(x0, y0, x1, y1, length = .07, angle = 20, col = col, lwd = lwd)
}

mini_key <- function(x, y, labels = arch_label, colours = pal[arch],
                     codes = arch_code, cex = .55, gap = .18) {
  for (i in seq_along(labels)) {
    xx <- x + ((i - 1) %% 3) * gap
    yy <- y - floor((i - 1) / 3) * .045
    points(xx, yy, pch = 21, bg = colours[i], col = white, cex = 1.25)
    text(xx + .018, yy, paste0(codes[i], "  ", labels[i]), adj = c(0, .5), cex = cex)
  }
}

heat_colour <- function(z, lim = max(abs(z), na.rm = TRUE)) {
  ramp <- colorRampPalette(c(pal["feedforward"], "#F7F3EA", pal["unlimited_shared"]))
  cols <- ramp(101)
  idx <- pmax(1, pmin(101, round((z + lim) / (2 * lim) * 100) + 1))
  cols[idx]
}

draw_heatmap <- function(mat, row_labels, col_labels, show_values = TRUE,
                         lim = max(abs(mat), na.rm = TRUE), cex = .60) {
  nr <- nrow(mat); nc <- ncol(mat)
  plot.new(); plot.window(c(0, nc), c(0, nr), xaxs = "i", yaxs = "i")
  for (r in seq_len(nr)) for (cc in seq_len(nc)) {
    val <- mat[r, cc]
    rect(cc - 1, nr - r, cc, nr - r + 1, col = heat_colour(val, lim), border = white)
    if (show_values) text(cc - .5, nr - r + .5, sprintf("%+.2f", val),
                          cex = cex, col = if (abs(val) > .62 * lim) white else ink)
  }
  axis(2, at = rev(seq_len(nr) - .5), labels = row_labels, tick = FALSE,
       line = -.35, cex.axis = .62)
  axis(1, at = seq_len(nc) - .5, labels = col_labels, tick = FALSE,
       line = -.55, cex.axis = .62, las = 2)
  box(col = white)
}

heatmap_canvas <- function(mat, row_labels, col_labels, x0 = .22, y0 = .16,
                           x1 = .94, y1 = .84, lim = max(abs(mat), na.rm = TRUE),
                           cex = .55, show_scale = TRUE) {
  nr <- nrow(mat); nc <- ncol(mat)
  cw <- (x1 - x0) / nc; ch <- (y1 - y0) / nr
  for (r in seq_len(nr)) for (cc in seq_len(nc)) {
    xa <- x0 + (cc - 1) * cw; ya <- y1 - r * ch
    val <- mat[r, cc]
    rect(xa, ya, xa + cw, ya + ch, col = heat_colour(val, lim), border = white)
    text(xa + cw / 2, ya + ch / 2, sprintf("%+.2f", val), cex = cex,
         col = if (abs(val) > .62 * lim) white else ink)
  }
  for (r in seq_len(nr)) text(x0 - .018, y1 - (r - .5) * ch, row_labels[r],
                               adj = 1, cex = cex, col = ink)
  for (cc in seq_len(nc)) text(x0 + (cc - .5) * cw, y0 - .025, col_labels[cc],
                                srt = 45, adj = c(1, 1), cex = cex, col = ink)
  if (show_scale) {
    sx0 <- x1 + .014; sx1 <- x1 + .031
    sy <- seq(y0, y1, length.out = 52)
    z <- seq(-lim, lim, length.out = 51)
    for (i in seq_along(z)) rect(sx0, sy[i], sx1, sy[i + 1],
                                 col = heat_colour(z[i], lim), border = NA)
    rect(sx0, y0, sx1, y1, border = light_ink, lwd = .6)
    text(sx1 + .009, y0, sprintf("%+.2f", -lim), adj = c(0, .5), cex = cex * .72)
    text(sx1 + .009, (y0 + y1) / 2, "0", adj = c(0, .5), cex = cex * .72)
    text(sx1 + .009, y1, sprintf("%+.2f", lim), adj = c(0, .5), cex = cex * .72)
  }
}

mock_data <- function() {
  set.seed(497987)
  seeds <- expand.grid(architecture = arch, seed = 1:20, stringsAsFactors = FALSE)
  centre <- c(.43, .49, .61, .46, .47)
  seeds$rms <- centre[match(seeds$architecture, arch)] + rnorm(nrow(seeds), 0, .045)
  seeds$cosine <- pmin(.98, .61 + (1 - seeds$rms) * .65 + rnorm(nrow(seeds), 0, .055))
  seeds$target_drop <- c(.27, .16, .045, .21, .01)[match(seeds$architecture, arch)] +
    rnorm(nrow(seeds), 0, .025)
  seeds$random_drop <- c(.078, .048, .020, .055, .004)[match(seeds$architecture, arch)] +
    rnorm(nrow(seeds), 0, .010)
  metric <- matrix(c(
    .02, .03, -.04, .01,
    .01, .01, -.01, .00,
    -.01, .02, -.03, .01,
    .01, .03, .02, .01,
    .00, .02, -.01, .00
  ), nrow = 5, byrow = TRUE)
  human <- matrix(c(
    .16, .03, .02, -.04,
    .10, .04, .01, -.01,
    .13, .02, .03, -.02,
    .06, .01, -.05, -.04,
    -.15, .00, -.08, -.06
  ), nrow = 5, byrow = TRUE)
  stages <- expand.grid(architecture = arch, seed = 1:20,
                        stage = c("Task trained", "Human adapted", "Sham"),
                        stringsAsFactors = FALSE)
  base <- c(1.17, 1.34, 1.27, 1.40, 1.38)
  shift <- c("Task trained" = 0, "Human adapted" = -.14, "Sham" = -.04)
  stages$distance <- base[match(stages$architecture, arch)] + shift[stages$stage] +
    rnorm(nrow(stages), 0, .035)
  list(seeds = seeds, metric = metric, human = human, stages = stages)
}

dat <- mock_data()

fig1 <- function(stamp = mock_mode) {
  layout(matrix(c(1, 1, 2, 3), 2, 2, byrow = TRUE), heights = c(1.05, .82))
  par(oma = c(.45, .35, 2.55, .30), mar = c(.15, .15, .15, .15))
  canvas(); panel_label("a", "One study, linked analytical stages", y = .95)
  round_box(.04, .62, .22, .83, white, pal["feedforward"])
  text(.13, .75, "HUMAN EEG", font = 2, cex = .74, col = pal["feedforward"])
  text(.13, .685, "Discovery + 2 replications", cex = .67)
  round_box(.04, .30, .22, .51, white, pal["constrained_shared"])
  text(.13, .43, "5 MACHINE MODELS", font = 2, cex = .74, col = pal["constrained_shared"])
  text(.13, .365, "20 independent seeds each", cex = .67)
  round_box(.31, .47, .48, .68, panel_fill, mid_ink)
  text(.395, .60, "COMMON", font = 2, cex = .70)
  text(.395, .54, "4-metric geometry", cex = .67)
  arrow_link(.22, .72, .31, .61, pal["feedforward"])
  arrow_link(.22, .40, .31, .54, pal["constrained_shared"])
  round_box(.57, .55, .73, .79, white, pal["private_modules"])
  text(.65, .72, "CHARACTERIZE", font = 2, cex = .68, col = pal["private_modules"])
  text(.65, .65, "Characterize", cex = .67)
  text(.65, .60, "Rank theories", cex = .67)
  round_box(.79, .55, .96, .79, white, pal["unlimited_shared"])
  text(.875, .72, "ADAPT + TEST", font = 2, cex = .68, col = pal["unlimited_shared"])
  text(.875, .65, "Adapt models", cex = .67)
  text(.875, .60, "Test mechanism", cex = .67)
  arrow_link(.48, .575, .57, .67)
  arrow_link(.73, .67, .79, .67, pal["unlimited_shared"])
  round_box(.57, .25, .96, .43, white, light_ink)
  text(.59, .365, "INDEPENDENT ARMS", font = 2, cex = .61, col = mid_ink, adj = 0)
  text(.59, .305, "Human data never train baseline models; adaptation is isolated in its own analysis stage.",
       cex = .61, adj = 0)
  text(.59, .265, "All headline geometry is evaluated on held-out subjects and seeds.", cex = .61, adj = 0)

  canvas(); panel_label("b", "Falsifiable architecture predictions", y = .95)
  mat <- matrix(c(.05, -.05, -.12, -.05, .08, .02, -.05, .00,
                  .10, .05, .12, .08, .18, .14, .18, .12, .12, .10, .08, .05),
                nrow = 5, byrow = TRUE)
  heatmap_canvas(mat, paste0(arch_code, "  ", arch_label),
                 c("Gain", "Broadcast", "Persistence", "Concentration"),
                 x0 = .30, y0 = .18, x1 = .89, y1 = .82, lim = .20, cex = .48)
  text(.055, .045, "Predeclared patterns; a losing prediction remains visible.", adj = 0,
       cex = .57, col = mid_ink)

  canvas(); panel_label("c", "Evidence ladder", y = .95)
  ys <- c(.76, .59, .42, .25)
  labs <- c("Discover", "Replicate", "Perturb", "Adapt")
  desc <- c("Gabor defines centre + scale", "Kronemer and Somato test transfer",
            "Targeted vs random subspace", "Human target vs matched sham")
  cols <- c(pal["feedforward"], pal["private_modules"], pal["constrained_shared"], pal["unlimited_shared"])
  for (i in 1:4) {
    points(.13, ys[i], pch = 21, bg = cols[i], col = white, cex = 2.1)
    text(.13, ys[i], i, cex = .55, font = 2, col = white)
    text(.21, ys[i] + .025, labs[i], adj = 0, font = 2, cex = .66)
    text(.21, ys[i] - .035, desc[i], adj = 0, cex = .57, col = mid_ink)
    if (i < 4) segments(.13, ys[i] - .045, .13, ys[i + 1] + .045, col = light_ink, lwd = 2)
  }
  figure_title("A common geometry links independent brains and machines",
               "Characterization identifies the signature; adaptation tests whether models acquire it causally")
  if (stamp) mock_stamp()
}

fig2 <- function(stamp = mock_mode) {
  layout(matrix(c(1, 1, 2, 3, 4, 4), 2, 3, byrow = TRUE), widths = c(1, 1, 1))
  par(oma = c(.5, .35, 2.55, .3), mar = c(3.2, 3.5, 1.7, .5))
  canvas(); panel_label("a", "Five computational claims, one controlled task")
  mini_key(.05, .73, cex = .60, gap = .28)
  for (i in 1:5) {
    x <- .08 + (i - 1) * .18
    round_box(x - .055, .25, x + .055, .48, white, pal[i])
    points(x + c(-.025, .025), c(.38, .38), pch = 21, bg = pal[i], col = white, cex = 1.25)
    if (i == 1) arrow_link(x - .018, .32, x + .018, .32, pal[i])
    if (i == 2) segments(x - .025, .31, x - .025, .37, col = pal[i], lwd = 2)
    if (i >= 3) arrows(x + .025, .35, x - .025, .35, length = .05, col = pal[i])
    if (i >= 4) symbols(x, .28, circles = if (i == 4) .018 else .030,
                         inches = FALSE, add = TRUE, bg = alpha(pal[i], .35), fg = pal[i])
  }
  text(.05, .12, "Architecture varies; training objective, stimulus statistics, accuracy target, seeds, and evaluation are matched.",
       adj = 0, cex = .62, col = mid_ink)

  x <- 0:10; y <- plogis((x - 3.2) * 2) * .34 + .66
  plot(x, y, type = "l", lwd = 2.4, col = pal["unlimited_shared"], ylim = c(.64, 1.01),
       xlab = "Difficulty bin", ylab = "Presence accuracy", bty = "l")
  panel_label("b", "Accuracy is matched")
  points(x, y, pch = 21, bg = pal["unlimited_shared"], col = white)
  abline(v = 3, lty = 3, col = mid_ink)
  text(3.2, .70, "matched threshold", adj = 0, cex = .55, col = mid_ink)

  dd <- dat$seeds
  plot(c(-.02, .34), c(.5, 5.5), type = "n", axes = FALSE,
       xlab = "Accuracy drop", ylab = "")
  panel_label("c", "Causal subspaces")
  axis(1); axis(2, at = 1:5, labels = arch_code, tick = FALSE, cex.axis = .68)
  abline(v = 0, lty = 3, col = light_ink)
  for (i in 1:5) {
    z <- dd[dd$architecture == arch[i],]
    segments(z$random_drop, i, z$target_drop, i, col = alpha(pal[i], .18), lwd = .7)
    points(z$random_drop, rep(i, 20), pch = 1, col = alpha(mid_ink, .45), cex = .55)
    points(z$target_drop, rep(i, 20), pch = 21, bg = alpha(pal[i], .55), col = white, cex = .65)
    m <- ci(z$target_drop); points(m[1], i, pch = 18, col = pal[i], cex = 1.25)
  }
  legend("bottomright", c("Targeted", "Random"), pch = c(21, 1),
         pt.bg = c(ink, NA), col = c(ink, mid_ink), bty = "n", cex = .55)

  canvas(); panel_label("d", "Accuracy-matched machine geometry")
  heatmap_canvas(dat$metric, paste0(arch_code, "  ", arch_label),
                 c("Gain", "Broadcast", "Persistence", "Concentration"),
                 x0 = .22, y0 = .16, x1 = .64, y1 = .84, lim = .05, cex = .53)
  text(.73, .70, "Causal relevance is established", adj = 0, cex = .66, col = mid_ink)
  text(.73, .63, "before human-machine comparison.", adj = 0, cex = .66, col = mid_ink)
  round_box(.72, .26, .96, .52, white, pal["control"])
  text(.75, .45, "CONTROLLED", adj = 0, font = 2, cex = .58, col = pal["unlimited_shared"])
  text(.75, .37, "Equal performance + seeds", adj = 0, cex = .60)
  figure_title("Controlled machine models establish the causal comparison space",
               "Theory is compared only after performance matching and targeted-subspace validation")
  if (stamp) mock_stamp()
}

fig3 <- function(stamp = mock_mode) {
  layout(matrix(c(1, 1, 2, 3, 4, 5), 2, 3, byrow = TRUE))
  par(oma = c(.5, .35, 2.55, .3), mar = c(3.1, 3.6, 1.7, .6))
  canvas(); panel_label("a", "Discovery signature and temporal generalization")
  cols <- c(pal["feedforward"], pal["private_modules"], pal["constrained_shared"], pal["unlimited_shared"])
  t <- seq(-250, 700, length.out = 110)
  for (i in 1:4) {
    yy <- .035 * exp(-((t - c(320, 80, 250, 420)[i]) / c(170, 200, 220, 250)[i])^2) * c(1, .6, .75, -.7)[i]
    xx <- (t - min(t)) / diff(range(t)) * .83 + .10
    base <- .74 - (i - 1) * .18
    polygon(c(xx, rev(xx)), c(base + yy - .018, rev(base + yy + .018)),
            col = alpha(cols[i], .16), border = NA)
    lines(xx, base + yy, col = cols[i], lwd = 2)
    segments(.10, base, .93, base, col = alpha(mid_ink, .35), lty = 3)
    text(.02, base, c("Gabor", "Kronemer early", "Kronemer late", "Somato")[i],
         adj = 0, cex = .57, col = mid_ink)
  }
  text(.10, .08, "-250", cex = .52); text(.93, .08, "700 ms", adj = 1, cex = .52)
  segments(.48, .15, .48, .83, col = pal["control"], lty = 3, lwd = 1.5)
  segments(.65, .15, .65, .83, col = pal["control"], lty = 3, lwd = 1.5)
  text(.565, .88, "primary windows", cex = .55, col = mid_ink)

  vals <- c(.045, .012, .018, -.052, -.165); err <- c(.018, .012, .014, .027, .055)
  plot(c(-.24, .09), c(.5, 5.5), type = "n", axes = FALSE,
       xlab = "Access Index contrast (95% CI)", ylab = "")
  panel_label("b", "Participant-level effects")
  axis(1); axis(2, at = 1:5, labels = c("Gabor", "Kronemer early", "Kronemer late", "Somato early", "Somato late"),
       tick = FALSE, cex.axis = .58)
  abline(v = 0, col = mid_ink, lty = 3)
  dscol <- c(pal[1], pal[2], pal[2], pal[4], pal[4])
  segments(vals - err, 1:5, vals + err, 1:5, col = dscol, lwd = 2)
  points(vals, 1:5, pch = 21, bg = dscol, col = white, cex = 1.25)

  canvas(); panel_label("c", "Four-metric replication")
  heatmap_canvas(dat$human, c("Gabor", "Kronemer E", "Kronemer L", "Somato E", "Somato L"),
                 c("Gain", "Broadcast", "Persistence", "Concentration"),
                 x0 = .25, y0 = .18, x1 = .88, y1 = .82, lim = .18, cex = .48)

  dist <- c(0, .41, .72, 1.10, 2.16); agree <- c(1, .76, .75, .76, .51)
  plot(dist, agree, xlim = c(-.08, 2.35), ylim = c(.42, 1.04),
       xlab = "Distance from Gabor signature", ylab = "Direction agreement", bty = "l")
  panel_label("d", "Discovery distance")
  points(dist, agree, pch = 21, bg = dscol, col = white, cex = 1.35)
  text(dist + .06, agree, c("G", "KE", "KL", "SE", "SL"), adj = 0, cex = .55)

  canvas(); panel_label("e", "Replication claim")
  round_box(.06, .56, .94, .82, white, pal["private_modules"])
  text(.10, .74, "WHAT REPLICATES", adj = 0, font = 2, cex = .60, col = pal["private_modules"])
  text(.10, .66, "Direction and multimetric structure", adj = 0, cex = .66)
  round_box(.06, .22, .94, .48, white, pal["unlimited_shared"])
  text(.10, .40, "WHAT MAY DIFFER", adj = 0, font = 2, cex = .60, col = pal["unlimited_shared"])
  text(.10, .32, "Latency, magnitude, and modality", adj = 0, cex = .66)
  figure_title("A human access geometry generalizes across heterogeneous datasets",
               "The discovery dataset defines the scale; replications remain separate and visibly testable")
  if (stamp) mock_stamp()
}

fig4 <- function(stamp = mock_mode) {
  layout(matrix(c(1, 1, 2, 3, 4, 4), 2, 3, byrow = TRUE))
  par(oma = c(.5, .35, 2.55, .3), mar = c(3.2, 3.7, 1.7, .6))
  
  x <- 1:4
  plot(c(.6, 4.4), c(-.12, .19), type = "n", axes = FALSE,
       xlab = "", ylab = "Standardized contrast")
  panel_label("a", "Human target versus machine profiles")
  axis(1, at = x, labels = c("Gain", "Broadcast", "Persistence", "Concentration"), tick = FALSE)
  axis(2); abline(h = 0, lty = 3, col = light_ink)
  human_target <- c(.16, .03, .015, -.035)
  lines(x, human_target, lwd = 3.5, col = ink); points(x, human_target, pch = 21, bg = ink, col = white, cex = 1.1)
  for (i in 1:5) {
    lines(x, dat$metric[i,], col = alpha(pal[i], .85), lwd = 1.8)
    points(x, dat$metric[i,], pch = 21, bg = pal[i], col = white, cex = .75)
  }
  text(1.05, .165, "Human target", adj = 0, cex = .56, font = 2, col = ink)
  legend("bottom", legend = c("Human", paste0(arch_code, "  ", arch_label)),
         col = c(ink, pal[arch]), lty = 1, lwd = c(2.8, rep(1.5, 5)),
         pch = 21, pt.bg = c(ink, pal[arch]), ncol = 3, bty = "n", cex = .44)

  dd <- dat$seeds
  plot(c(.30, .72), c(.5, 5.5), type = "n", axes = FALSE,
       xlab = "Human-machine RMS distance", ylab = "")
  panel_label("b", "Seed-level theory ranking")
  axis(1); axis(2, at = 1:5, labels = paste0(arch_code, "  ", arch_label), tick = FALSE, cex.axis = .58)
  for (i in 1:5) {
    z <- dd$rms[dd$architecture == arch[i]]
    points(z, jitter(rep(i, length(z)), amount = .10), pch = 21,
           bg = alpha(pal[i], .38), col = white, cex = .62)
    m <- ci(z); segments(m[1] - m[2], i, m[1] + m[2], i, col = pal[i], lwd = 2.4)
    points(m[1], i, pch = 18, col = pal[i], cex = 1.35)
  }

  plot(dd$rms, dd$cosine, pch = 21, bg = alpha(pal[match(dd$architecture, arch)], .48),
       col = white, xlab = "RMS distance", ylab = "Cosine similarity", bty = "l",
       ylim = c(.74, .995))
  panel_label("c", "Concordance, not distance alone")
  abline(v = median(dd$rms), h = median(dd$cosine), lty = 3, col = light_ink)
  text(.34, .96, "closer + aligned", adj = 0, cex = .56, col = mid_ink)

  canvas(); panel_label("d", "Capacity-limit falsification test")
  metrics <- c("Gain", "Broadcast", "Persistence", "Concentration")
  est <- c(.01, -.06, .10, .02); er <- c(.02, .05, .18, .02)
  mapx <- function(z) .23 + (z + .25) / .55 * .40
  segments(mapx(-.20), .17, mapx(-.20), .77, col = pal["private_modules"], lty = 3, lwd = 1.4)
  segments(mapx(.20), .17, mapx(.20), .77, col = pal["private_modules"], lty = 3, lwd = 1.4)
  segments(mapx(0), .17, mapx(0), .77, col = mid_ink)
  for (i in 1:4) {
    yy <- .25 + (i - 1) * .14
    text(.20, yy, metrics[i], adj = 1, cex = .58)
    segments(mapx(est[i] - er[i]), yy, mapx(est[i] + er[i]), yy, lwd = 2, col = ink)
    points(mapx(est[i]), yy, pch = 21, bg = ink, col = white, cex = 1.1)
  }
  text(mapx(-.20), .11, "-0.20", cex = .51); text(mapx(0), .11, "0", cex = .51); text(mapx(.20), .11, "+0.20", cex = .51)
  round_box(.69, .16, .95, .70, white, pal["control"])
  text(.72, .62, "DECISION RULE", adj = 0, font = 2, cex = .60, col = pal["unlimited_shared"])
  text(.72, .52, "Capacity limit wins only if", adj = 0, cex = .59)
  text(.72, .44, "the constrained model is", adj = 0, cex = .59)
  text(.72, .36, "reliably closer than the", adj = 0, cex = .59)
  text(.72, .28, "unlimited shared state.", adj = 0, cex = .59)
  figure_title("Human geometry adjudicates among competing computational claims",
               "All architectures can lose; the capacity account has an explicit equivalence-based falsifier")
  if (stamp) mock_stamp()
}

fig5 <- function(stamp = mock_mode) {
  layout(matrix(c(1, 1, 1, 2, 3, 4), 2, 3, byrow = TRUE))
  par(oma = c(.5, .35, 2.55, .3), mar = c(3.2, 3.8, 1.7, .6))
  canvas(); panel_label("a", "Anti-circular adaptation design")
  boxes <- list(c(.04,.46,.20,.70), c(.27,.46,.43,.70), c(.51,.56,.67,.78), c(.51,.30,.67,.50), c(.76,.40,.96,.68))
  fills <- rep(white, 5)
  borders <- c(pal[1], mid_ink, pal[4], pal[4], mid_ink)
  for (i in 1:5) round_box(boxes[[i]][1],boxes[[i]][2],boxes[[i]][3],boxes[[i]][4],fills[i],borders[i])
  text(.12,.61,"TRAIN EEG",font=2,cex=.66,col=pal[1]); text(.12,.53,"temporal target",cex=.59)
  text(.35,.61,"TASK MODEL",font=2,cex=.66); text(.35,.53,"frozen checkpoint",cex=.59)
  text(.59,.69,"HUMAN",font=2,cex=.64,col=pal[4]); text(.59,.62,"adaptation",cex=.58)
  text(.59,.42,"MATCHED",font=2,cex=.64,col=pal[5]); text(.59,.35,"sham",cex=.58)
  text(.86,.58,"HELD-OUT",font=2,cex=.64); text(.86,.50,"geometry +",cex=.58); text(.86,.44,"causality",cex=.58)
  arrow_link(.20,.58,.27,.58,pal[1]); arrow_link(.43,.58,.51,.67,pal[4]); arrow_link(.43,.54,.51,.40,pal[4])
  arrow_link(.67,.67,.76,.58); arrow_link(.67,.40,.76,.48)
  rect(.28,.16,.84,.25,col=white,border=pal[6],lwd=1.1)
  text(.56,.205,"Evaluation metrics never enter loss or checkpoint selection",font=2,cex=.61,col=pal[5])
  mini_key(.05,.10,cex=.50,gap=.29)

  st <- dat$stages
  plot(c(.98,1.48),c(.5,5.5),type="n",axes=FALSE,xlab="Held-out RMS distance (lower is closer)",ylab="")
  panel_label("b", "Stage trajectory")
  axis(1); axis(2,at=1:5,labels=arch_code,tick=FALSE,cex.axis=.68)
  for(i in 1:5){
    z <- st[st$architecture==arch[i],]
    means <- tapply(z$distance,z$stage,mean)
    segments(means["Task trained"],i,means["Human adapted"],i,col=alpha(pal[i],.55),lwd=2)
    points(means["Task trained"],i,pch=21,bg=white,col=pal[i],cex=1.05)
    points(means["Human adapted"],i,pch=21,bg=pal[i],col=white,cex=1.15)
    points(means["Sham"],i,pch=22,bg=pal[6],col=white,cex=.95)
  }
  legend("bottomright",c("Task","Human adapted","Sham"),pch=c(21,21,22),pt.bg=c(white,ink,pal[6]),bty="n",cex=.52)

  gains <- c(.09,.045,.12,.08,.02); er <- c(.022,.018,.025,.021,.020)
  plot(c(-.025,.16),c(.5,5.5),type="n",axes=FALSE,xlab="Adapted gain - sham gain",ylab="")
  panel_label("c", "Human-specific gain")
  axis(1); axis(2,at=1:5,labels=arch_code,tick=FALSE)
  abline(v=0,lty=3,col=mid_ink)
  segments(gains-er,1:5,gains+er,1:5,col=pal[arch],lwd=2)
  points(gains,1:5,pch=21,bg=pal[arch],col=white,cex=1.2)

  canvas(); panel_label("d", "Acquired geometry")
  adapted <- matrix(c(.25,.18,.11,.20,.15,.09,.07,.05,.16,.13,.17,.15,.17,.25,.14,.17,.11,.11,.04,.05),nrow=5,byrow=TRUE)
  heatmap_canvas(adapted,arch_code,c("Gain","Broadcast","Persistence","Concentration"),
                 x0=.20,y0=.18,x1=.88,y1=.82,lim=.25,cex=.50)
  figure_title("Human-neural adaptation tests whether the geometry is acquirable",
               "Matched sham training separates target-specific acquisition from generic optimization")
  if (stamp) mock_stamp()
}

fig6 <- function(stamp = mock_mode) {
  layout(matrix(c(1,2,3,4,5,5),2,3,byrow=TRUE))
  par(oma=c(.5,.35,2.55,.3),mar=c(3.2,3.8,1.7,.6))
  x <- c(.013,.019,.017,.023,.025); y <- c(.11,.075,.16,.115,.06)
  plot(x,y,xlim=c(.009,.029),ylim=c(0,.19),xlab="Relative parameter displacement",ylab="Held-out gain",bty="l")
  panel_label("a","Efficiency frontier")
  abline(h=0,lty=3,col=light_ink)
  for(i in 1:5){points(x[i]+rnorm(16,0,.0015),y[i]+rnorm(16,0,.018),pch=16,col=alpha(pal[i],.28),cex=.55);points(x[i],y[i],pch=21,bg=pal[i],col=white,cex=1.35)}
  text(.010,.18,"more gain / less change",adj=0,cex=.54,col=mid_ink)

  est <- c(-.0005,-.0004,-.0006,-.0015,-.0012); er <- rep(.001,5)
  plot(c(-.022,.022),c(.5,5.5),type="n",axes=FALSE,xlab="Presence accuracy change",ylab="")
  panel_label("b","Task retention")
  segments(-.02,.5,-.02,5.5,col=pal[6],lty=3,lwd=1.3)
  segments(.02,.5,.02,5.5,col=pal[6],lty=3,lwd=1.3); abline(v=0,col=mid_ink)
  axis(1);axis(2,at=1:5,labels=arch_code,tick=FALSE)
  segments(est-er,1:5,est+er,1:5,col=pal[arch],lwd=2);points(est,1:5,pch=21,bg=pal[arch],col=white,cex=1.15)
  text(.019,.78,"prespecified gate",adj=1,cex=.52,col=mid_ink)

  before <- c(.055,.030,.065,.047,.043); after <- before+c(.022,.015,.025,.026,.012)
  plot(c(.5,5.5),c(0,.105),type="n",axes=FALSE,xlab="Architecture",ylab="Targeted drop - random drop")
  panel_label("c","Causal specificity")
  axis(1,at=1:5,labels=arch_code,tick=FALSE);axis(2);abline(h=0,lty=3,col=light_ink)
  for(i in 1:5){segments(i-.10,before[i],i+.10,after[i],col=alpha(pal[i],.6),lwd=1.5);points(i-.10,before[i],pch=21,bg=white,col=pal[i]);points(i+.10,after[i],pch=21,bg=pal[i],col=white)}
  legend("topleft",c("Task","Adapted"),pch=21,pt.bg=c(white,ink),bty="n",cex=.52)

  canvas();panel_label("d","Transfer beyond discovery")
  trans <- matrix(c(.10,.06,.09,.02,.10,.05,.08,.01,.12,.07,.11,.02,.11,.06,.10,.02,.10,.04,.08,0),nrow=5,byrow=TRUE)
  heatmap_canvas(trans,arch_code,c("Kronemer E","Kronemer L","Somato E","Somato L"),
                 x0=.20,y0=.18,x1=.88,y1=.82,lim=.12,cex=.48)

  canvas();panel_label("e","One-study synthesis")
  round_box(.05,.48,.31,.78,white,pal[1]);round_box(.37,.48,.63,.78,white,pal[4]);round_box(.69,.48,.95,.78,white,pal[5])
  text(.18,.70,"CHARACTERIZE",font=2,cex=.66,col=pal[1]);text(.18,.61,"Which architecture",cex=.62);text(.18,.55,"already resembles humans?",cex=.62)
  text(.50,.70,"ADAPT + TEST",font=2,cex=.66,col=pal[4]);text(.50,.61,"Which architecture can",cex=.62);text(.50,.55,"acquire the geometry?",cex=.62)
  text(.82,.70,"JOINT CLAIM",font=2,cex=.66,col=pal[5]);text(.82,.61,"Similarity + acquisition",cex=.62);text(.82,.55,"+ causal preservation",cex=.62)
  arrow_link(.31,.63,.37,.63);arrow_link(.63,.63,.69,.63)
  text(.05,.28,"Strong support requires convergence across baseline fit, adaptation specificity, retention, causal intervention, and dataset transfer.",adj=0,cex=.67,col=mid_ink)
  mini_key(.08,.15,cex=.54,gap=.28)
  figure_title("Across the study | Resemblance is separated from mechanism",
               "No single panel is decisive; the article closes on a transparent, multicomponent evidence chain")
  if(stamp) mock_stamp()
}

figures <- list(
  "Fig1-unified-study-logic" = fig1,
  "Fig2-machine-foundation" = fig2,
  "Fig3-human-discovery-replication" = fig3,
  "Fig4-baseline-theory-competition" = fig4,
  "Fig5-human-neural-adaptation" = fig5,
  "Fig6-study-synthesis" = fig6
)

render_one <- function(stem, fun) {
  png(file.path(output_root, paste0(stem, "-mock.png")), width = 2400, height = 1650,
      res = 300, bg = paper, type = "cairo")
  set_theme(); fun(TRUE); dev.off()
  pdf(file.path(output_root, paste0(stem, "-mock.pdf")), width = 8.0, height = 5.5,
      bg = paper, useDingbats = FALSE)
  set_theme(); fun(TRUE); dev.off()
}

if (!mock_mode) {
  required_exp2 <- c("seed-stage-summary.csv", "sham-comparison.csv", "adaptation-cost.csv",
                     "post-adaptation-interventions.csv", "external-transfer.csv", "stage-geometry.csv")
  missing <- required_exp2[!file.exists(file.path(adaptation_root, required_exp2))]
  if (length(missing)) stop("Production rendering blocked; missing adaptation-stage tables: ",
                            paste(missing, collapse = ", "))
  stop("Production data adapter is intentionally locked until the adaptation-stage table schema is frozen. Use --mock for layout review.")
}

for (nm in names(figures)) render_one(nm, figures[[nm]])

book <- file.path(output_root, "unified-article-figure-mockbook.pdf")
pdf(book, width = 8.0, height = 5.5, bg = paper, useDingbats = FALSE, onefile = TRUE)
for (fun in figures) { set_theme(); fun(TRUE) }
dev.off()
cat("Rendered", length(figures), "unified mock figures to",
    normalizePath(output_root, winslash = "/", mustWork = TRUE), "\n")
