library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# Folder with the input tables (embeddings, dataset summaries, metrics summaries); set AINU_R_DATA or run from it.
setwd(Sys.getenv("AINU_R_DATA", "."))
dir.create("figures", showWarnings = FALSE)
df <- read_csv("patient_accuracy_mean_sd_for_R.csv", show_col_types = FALSE)

df <- df %>%
  mutate(patient_new_name = as.character(patient))  # patient already holds the paper label
# Convert to long format for plotting
plot_df <- df %>%
  select(
    patient,
    patient_new_name,
    best_seed,
    mean_old_accuracy,
    sd_old_accuracy,
    mean_new_accuracy,
    sd_new_accuracy
  ) %>%
  pivot_longer(
    cols = c(mean_old_accuracy, mean_new_accuracy),
    names_to = "model",
    values_to = "mean_accuracy"
  ) %>%
  mutate(
    sd_accuracy = case_when(
      model == "mean_old_accuracy" ~ sd_old_accuracy,
      model == "mean_new_accuracy" ~ sd_new_accuracy
    ),
    model = recode(
      model,
      "mean_old_accuracy" = "Old model",
      "mean_new_accuracy" = "New model"
    ),
    model = factor(model, levels = c("Old model", "New model")),
    sd_accuracy = ifelse(is.na(sd_accuracy), 0, sd_accuracy),
    mean_accuracy_pct = mean_accuracy,
    sd_accuracy_pct = sd_accuracy,
    patient = factor(patient, levels = unique(patient))
  )

# Plot
p <- ggplot(plot_df, aes(x = model, y = mean_accuracy_pct, fill = model)) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(
      ymin = mean_accuracy_pct - sd_accuracy_pct,
      ymax = mean_accuracy_pct + sd_accuracy_pct
    ),
    width = 0.2
  ) +
  facet_wrap(~ patient_new_name, scales = "fixed") +
  labs(
    x = NULL,
    y = "Mean Accuracy",
    fill = "Model",
    title = "Mean old vs new accuracy per patient",
    subtitle = "Best seed selected per patient; error bars show SD across 5 splits"
  ) +
  theme_bw() +
  #theme(
  #  strip.text = element_text(face = "bold"),
  #  axis.text.x = element_text(angle = 45, hjust = 1),
  #  legend.position = "none"
  #) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 14),
    axis.text.y = element_text(size = 14),
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 14),
    legend.background = element_rect(fill = "white", color = "grey70", linewidth = 0.5),
    plot.margin = margin(10, 10, 10, 10)
  )
p
ggsave("figures/retrain_frozen_model_onepatout.pdf", p, width = 4, height = 5, dpi = 300)




