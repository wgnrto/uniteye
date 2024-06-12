library("rstudioapi")
setwd(dirname(getActiveDocumentContext()$path))


library(devtools)
source_url("https://raw.githubusercontent.com/M-Colley/rCode/main/r_functionality.R")


main_df <- read_delim(file = "data.csv", delim = ";")
main_df <- as.data.frame(main_df)
names(main_df)

main_df$participant <- as.factor(main_df$participant)
main_df$condition <- as.factor(main_df$condition)
main_df$event <- as.factor(main_df$event)


levels(main_df$condition)[levels(main_df$condition) == "tobii"] <- "Tobii"
levels(main_df$condition)[levels(main_df$condition) == "uniteye"] <- "UnitEye"
levels(main_df$condition)[levels(main_df$condition) == "uniteye_filtered"] <- "UnitEye Smoothed"


main_df$area <- factor(ifelse(main_df$event %in% c("1", "3", "9", "11"), "corner", ifelse(main_df$event %in% c("2", "4", "5", "7", "8", "10"), "inner", "center")), levels = c("center", "inner", "corner"))



main_df$euclidean_cm <- main_df$euclidean / 141 * 2.54
main_df$rmss2s_cm <- main_df$rmss2s / 141 * 2.54
main_df$sd_x_cm <- main_df$sd_x / 141 * 2.54
main_df$sd_y_cm <- main_df$sd_y / 141 * 2.54
main_df$dist_x_cm <- main_df$dist_x / 141 * 2.54
main_df$dist_y_cm <- main_df$dist_y / 141 * 2.54




main_df |>
  group_by(condition, event) |>
  arrange(desc(euclidean_cm))




# Taken from: https://www.sciencedirect.com/science/article/pii/S2213398421001391
BA.plot <- function(m1, m2) {
  means <- (m1 + m2) / 2
  
  diffs <- m1 - m2
  mdiff <- mean(diffs)
  sddiff <- sd(diffs)
  
  # Positioning of labels on the lines
  
  xmax <- max(means)
  
  xmin <- min(means)
  
  x1 <- (xmax - xmin) * 0.10
  
  x2 <- xmax - x1
  
  # Adding space for legend
  
  ylimh <- mdiff + (3.5 * sddiff)
  
  yliml <- mdiff - (3 * sddiff)
  
  # Plot data
  
  plot(diffs ~ means, xlab = "Average of two methods", ylab = "Difference of two methods", ylim = c(yliml, ylimh))
  
  abline(h = mdiff, col = "blue", lty = 2)
  
  text(x2, mdiff, "Mean")
  
  abline(h = 0)
  
  # Standard deviations lines & legend
  
  abline(h = mdiff + 1.96 * sddiff, col = "blue", lty = 2)
  
  text(x2, mdiff + 1.96 * sddiff, "Upper Limit")
  
  abline(h = mdiff - 1.96 * sddiff, col = "blue", lty = 2)
  
  text(x2, mdiff - 1.96 * sddiff, "Lower Limit")
  
  paramX <- round(mdiff, 2)
  paramY <- round(sddiff, 2)
  
  UL <- mdiff + 1.96 * sddiff
  UL <- round(UL, 2)
  
  LL <- mdiff - 1.96 * sddiff
  LL <- round(LL, 2)
  
  expr <- vector("expression", 4)
  
  expr[[1]] <- bquote(Mean[diff] == .(paramX))
  
  expr[[2]] <- bquote(SD[diff] == .(paramY))
  
  expr[[3]] <- bquote(LL == .(LL))
  
  expr[[4]] <- bquote(UL == .(UL))
  
  legend("topright", bty = "n", legend = expr)
}



# 1 pixel is 0.02 degree --> cm / 2.54 * 141 * 0.0172

# main_df$rmss2s
#
# main_df |> group_by(participant, event, condition) |> count() |> print(n=200)
#

# 1 Pixel = 0.018 cm
# 55.56 pixel = 1 cm
# 1 pixel = 0.0172 degree
# 1 cm = 0.956 degree
scaling_factor <- 0.956

#### euclidean ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "euclidean", factor_1 = "condition", factor_2 = "event")

modelArt <- art(euclidean ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "euclidean distance")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = euclidean_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("Euclidean distance (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Euclidean distance (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "Euclidean distance (in visual °)")
  )
ggsave("plots/euclidean.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



#### euclidean based on areas center, inner, outer ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "euclidean", factor_1 = "condition", factor_2 = "area")

modelArt <- art(euclidean ~ condition * area + Error(participant / (condition * area)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "euclidean distance")

rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 66 * 12)


main_df %>% ggplot() +
  aes(x = condition, y = euclidean_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("Euclidean distance (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Points", colour = "Validation Points") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Euclidean distance (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "Euclidean distance (in visual °)")
  )
ggsave("plots/euclidean_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



#### RMS-S2S ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "rmss2s", factor_1 = "condition", factor_2 = "event")

modelArt <- art(rmss2s ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "RMS-S2S")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = rmss2s_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("RMS-S2S (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "RMS-S2S (in cm)",
    sec.axis = sec_axis(trans = ~ . * scaling_factor, name = "RMS-S2S (in visual °)") # 141 PPI / inch= 2.54cm
  )
ggsave("plots/rmss2s.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)






checkAssumptionsForAnovaTwoFactors(data = main_df, y = "rmss2s", factor_1 = "condition", factor_2 = "area")

modelArt <- art(rmss2s ~ condition * area + Error(participant / (condition * area)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "RMS-S2S")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 66 * 12)
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 66 * 12)


main_df %>% ggplot() +
  aes(x = condition, y = rmss2s_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("RMS-S2S (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Areas", colour = "Validation Areas") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "RMS-S2S (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "RMS-S2S (in visual °)")
  )
ggsave("plots/rmss2s_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)





#### sd_x ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "sd_x", factor_1 = "condition", factor_2 = "event")

modelArt <- art(sd_x ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "SD in x direction")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = sd_x_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("SD x direction (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "SD x direction (in cm)",
    sec.axis = sec_axis(trans = ~ . * scaling_factor, name = "SD x direction (in visual °)") # 141 PPI / inch= 2.54cm
  )
ggsave("plots/sd_x.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




checkAssumptionsForAnovaTwoFactors(data = main_df, y = "sd_x", factor_1 = "condition", factor_2 = "area")

modelArt <- art(sd_x ~ condition * area + Error(participant / (condition * area)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "SD in x direction")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 66 * 12)
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 66 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = sd_x_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("SD x direction (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Areas", colour = "Validation Areas") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "SD x direction (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "SD x direction (in visual °)")
  )
ggsave("plots/sd_x_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)






#### sd_y ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "sd_y", factor_1 = "condition", factor_2 = "event")

modelArt <- art(sd_y ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "SD in y direction")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = sd_y_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("SD y direction (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "SD y direction (in cm)",
    sec.axis = sec_axis(trans = ~ . * scaling_factor, name = "SD y direction (in visual °)") # 141 PPI / inch= 2.54cm
  )
ggsave("plots/sd_y.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



checkAssumptionsForAnovaTwoFactors(data = main_df, y = "sd_y", factor_1 = "condition", factor_2 = "area")

modelArt <- art(sd_y ~ condition * area + Error(participant / (condition * area)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "SD in y direction")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 66 * 12)
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 66 * 12)


main_df %>% ggplot() +
  aes(x = condition, y = sd_y_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("SD y direction (in cm)") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Areas", colour = "Validation Areas") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "SD y direction (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "SD y direction (in visual °)")
  )
ggsave("plots/sd_y_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




#### dist_x ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "dist_x", factor_1 = "condition", factor_2 = "event")

modelArt <- art(dist_x ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "dist_x")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = dist_x_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Distance in x direction (in cm)",
    sec.axis = sec_axis(trans = ~ . * scaling_factor, name = "Distance in x direction (in visual °)") # 141 PPI / inch= 2.54cm
  )
ggsave("plots/dist_x.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




main_df %>% ggplot() +
  aes(x = condition, y = dist_x_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Areas", colour = "Validation Areas") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Distance in x direction (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "Distance in x direction (in visual °)")
  )
ggsave("plots/dist_x_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)





#### dist_y ####
checkAssumptionsForAnovaTwoFactors(data = main_df, y = "dist_y", factor_1 = "condition", factor_2 = "event")

modelArt <- art(dist_y ~ condition * event + Error(participant / (condition * event)), data = main_df) |> anova()
modelArt
reportART(modelArt, dv = "dist_y")

# condition - Tobii vs UnitEye
rFromNPAV(pvalue = modelArt$`Pr(>F)`[1], N = 44 * 12)
# validation point
rFromNPAV(pvalue = modelArt$`Pr(>F)`[2], N = 44 * 12)

main_df %>% ggplot() +
  aes(x = condition, y = dist_y_cm, fill = event, colour = event, group = event) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  ylab("dist_y") +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Point", colour = "Validation Point") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Distance in y direction (in cm)",
    sec.axis = sec_axis(trans = ~ . * scaling_factor, name = "Distance in y direction (in visual °)") # 141 PPI / inch= 2.54cm
  )
ggsave("plots/dist_y.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




main_df %>% ggplot() +
  aes(x = condition, y = dist_y_cm, fill = area, colour = area, group = area) +
  theme_lucid(axis.text.size = 20, axis.title.size = 20) +
  scale_color_see() +
  theme(legend.title = element_text(size = 18), axis.title = element_text(size = 18), axis.text = element_text(size = 18), plot.title = element_text(size = 28), plot.subtitle = element_text(size = 18), legend.background = element_blank(), legend.position = "inside", legend.position.inside = c(0.25, 0.65), legend.text = element_text(size = 16)) +
  labs(fill = "Validation Areas", colour = "Validation Areas") +
  xlab("") +
  stat_summary(fun = mean, geom = "point", size = 4.0, alpha = 0.9) +
  stat_summary(fun = mean, geom = "line", linewidth = 2, alpha = 0.5) +
  stat_summary(fun.data = "mean_cl_normal", geom = "errorbar", width = .5, position = position_dodge(width = 0.01), alpha = 0.5) +
  scale_y_continuous(
    name = "Distance in y direction (in cm)",
    sec.axis = sec_axis(transform = ~ . * scaling_factor, name = "Distance in y direction (in visual °)")
  )
ggsave("plots/dist_y_area.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)




# not available in the csv.

# mean(main_df$lux)
# sd(main_df$lux)





#### OPTIONAL - Bland-Altman Analysis ####

## see Bland, J. M., & Altman, D. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. The Lancet, 327(8476), 307-310.
library(BlandAltmanLeh)

main_df$rmss2s <- as.numeric(main_df$rmss2s)

test <- NULL
test$tobii <- main_df |>
  dplyr::filter(condition == "Tobii") |>
  pull(rmss2s) # Using pull to extract the vector directly

test$uniteye <- main_df |>
  dplyr::filter(condition == "UnitEye") |>
  dplyr::pull(rmss2s) 

test$uniteye_filtered <- main_df |>
  dplyr::filter(condition == "UnitEye Smoothed") |>
  dplyr::pull(rmss2s)



bland.altman.plot(test$tobi, test$uniteye, graph.sys = "ggplot2") +
  xlab("Mean value of the measurements for RMS-S2S") +
  ylab("Mean value difference for RMS-S2S") +
  ggtitle("Bland-Altman-Plot - Tobii vs UnitEye") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("plots/bland-altman-rmss2s-tobii-uniteye.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



bland.altman.plot(test$tobi, test$uniteye_filtered, graph.sys = "ggplot2") +
  xlab("Mean value of the measurements for RMS-S2S") +
  ylab("Mean value difference for RMS-S2S") +
  ggtitle("Bland-Altman-Plot- Tobii vs UnitEye Smoothed") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("plots/bland-altman-rmss2s-tobii-uniteye-smoothed.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)


# bland.altman.plot(test$tobi, test$uniteye_filtered, sunflower = TRUE)

# see: https://www.sciencedirect.com/science/article/pii/S2213398421001391
BA.plot(test$tobi, test$uniteye_filtered)
ggsave("plots/bland-altman-rmss2s-tobii-uniteye-smoothed_BAPlot.pdf", width = pdfwidth, height = pdfheight + 2, device = cairo_pdf)



#### Demographic ####


demo_df <- read_xlsx(path = "uniteye-technical-demographic.xlsx")
demo_df <- as.data.frame(demo_df)
names(demo_df)

report::report_participants(demo_df)

# table(demo_df$gender)

# A2 Student (Uni); A3 Angestellter
table(demo_df$job)
table(demo_df$Augenfarbe)
# gbrow = green with a brown peripupilary ring
table(demo_df$AugenfarbeDetail)
table(demo_df$Makeup)
table(demo_df$Ethnicity)
table(demo_df$Brille)
table(demo_df$Kontaktlinsen)
table(demo_df$G01Q12) # Haarfarbe, translated manually to english
# table(demo_df$eyeshape) # added manually --> will be a sketch
