library(ggplot2)
library(dplyr)
library(ggrepel)
library(readr)
library(ggtext)
library(ggsci)  # install if needed: install.packages("ggsci")
library(patchwork)
library(rstatix)
library(ggpubr)
library(tidyr)
library(stringr)
library(viridis)
library(readxl)
library(purrr)

# Folder with the input tables (embeddings, dataset summaries, metrics summaries); set AINU_R_DATA or run from it.
setwd(Sys.getenv("AINU_R_DATA", "."))
dir.create("figures", showWarnings = FALSE)
#-----------------------------------------
# MDS
#-----------------------------------------

mds1 <- read.csv("embeddings/mds/embeddings_preleuk_sample4s1_augmentation.csv", header = TRUE, sep = ",")
mds2 <- read.csv("embeddings/mds/embeddings_preleuk_sample4s2_augmentation.csv", header = TRUE, sep = ",")
mds3 <- read.csv("embeddings/mds/embeddings_preleuk_sample4s3_augmentation.csv", header = TRUE, sep = ",")
mds4 <- read.csv("embeddings/mds/embeddings_preleuk_sample4s4_augmentation.csv", header = TRUE, sep = ",")
mds5 <- read.csv("embeddings/mds/embeddings_preleuk_sample4s5_augmentation.csv", header = TRUE, sep = ",")

data <- read_tsv("dataset_summary_MDS.txt", show_col_types = FALSE)

df_original <- list(
  split1 = mds1,
  split2 = mds2,
  split3 = mds3,
  split4 = mds4,
  split5 = mds5
) %>%
  imap(function(df, split_name) {
    df %>%
      select(File, celltype, condition, prob_positive) %>%
      rename(!!paste0("prob_positive_", split_name) := prob_positive)
  }) %>%
  reduce(full_join, by = c("File", "celltype", "condition"))

df <- df_original %>%
  left_join(
    data %>% select(patientID, patient),
    by = c("celltype" = "patientID")
  )

df %>%     # check if I have NA values
  filter(is.na(patient))


# avg splits 
df <- df %>%
  rowwise() %>%
  mutate(
    avg_prob_positive = mean(
      c_across(starts_with("prob_positive_split")),
      na.rm = TRUE
    ),
    sd_prob_positive = sd(
      c_across(starts_with("prob_positive_split")),
      na.rm = TRUE
    ),
    avg_prob_positive = ifelse(is.nan(avg_prob_positive), NA, avg_prob_positive),
    sd_prob_positive = ifelse(is.na(sd_prob_positive), NA, sd_prob_positive)
  ) %>%
  ungroup()


# plot %

df_percent_bins <- df %>%
  filter(!is.na(patient), !is.na(avg_prob_positive), !is.na(File)) %>%
  mutate(
    prob_bin = cut(
      avg_prob_positive,
      breaks = seq(0, 1, by = 0.1),
      include.lowest = TRUE,
      right = FALSE
    )
  ) %>%
  count(patient, prob_bin) %>%
  group_by(patient) %>%
  mutate(percent = 100 * n / sum(n)) %>%
  ungroup()

p <- ggplot(df_percent_bins, aes(x = prob_bin, y = percent)) +
  geom_col(fill = "#F4A582") +
  facet_wrap(~ patient) +
  scale_y_continuous(limits = c(0, 100)) +
  theme_bw() +
  labs(
    x = "probability of being AML",
    y = "Percentage of cells",
    title = "Percentage distribution of cells by prob_positive per patient (model n4, split 3)"
  ) +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
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
ggsave("figures/MDS_distribution_prob_pos_perc_over5splits_pink.pdf", p, width = 7, height = 6.5, dpi = 300)


summary_09 <- df %>%
  filter(!is.na(patient), !is.na(File), !is.na(avg_prob_positive)) %>%
  distinct(patient, File, .keep_all = TRUE) %>%   # count each cell/File once
  group_by(patient) %>%
  summarise(
    total_cells = n(),
    cells_gt_09 = sum(avg_prob_positive > 0.9),
    percent_gt_09 = 100 * cells_gt_09 / total_cells,
    .groups = "drop"
  )

summary_09
a <- ggplot(summary_09, aes(x = patient, y = percent_gt_09)) +
  geom_col(fill = "#F4A582") +
  scale_y_continuous(limits = c(0, 100)) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
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
  ) +
  labs(
    x = "Patient",
    y = "% cells with prob_positive > 0.9",
    title = "Percentage of highly positive cells per patient (Modle n4, split 3)"
  )
a
ggsave("figures/MDS_distribution_prob_pos_over90_over5splits_colored_new.pdf", a, width = 4, height = 3, dpi = 300)
write.table(
  summary_09,
  file = "figures/MDS_summary_09.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# plot mean probability with SD of only one patient 

patient_to_plot <- "MDS08"   # change this to the patient you want

df_one_patient <- df %>%
  filter(
    patient == patient_to_plot,
    !is.na(avg_prob_positive),
    !is.na(sd_prob_positive)
  ) %>%
  arrange(avg_prob_positive) %>%
  mutate(cell_index = row_number())

f <- ggplot(df_one_patient, aes(x = cell_index, y = avg_prob_positive)) +
  geom_point(size = 1) +
  geom_errorbar(aes(
      ymin = avg_prob_positive - sd_prob_positive,
      ymax = avg_prob_positive + sd_prob_positive),
      width = 0.2,
      linewidth = 0.3
  ) +
  theme_bw() +
  labs(
    x = "Signle cell",
    y = "Average AML probability",
    title = paste("Cell-level probability of being AML:", patient_to_plot)) +
  theme(
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
f
ggsave("figures/MDS_AML17_distribution_probabilities.pdf", f, width = 5, height = 3, dpi = 300)


# plot 5 values 
df_one_patient_long <- df %>%
  filter(
    patient == patient_to_plot,
    !is.na(avg_prob_positive)
  ) %>%
  arrange(avg_prob_positive) %>%
  mutate(
    File = factor(File, levels = File)   # keeps Files ordered by avg probability
  ) %>%
  select(
    File,
    patient,
    avg_prob_positive,
    starts_with("prob_positive_split")
  ) %>%
  pivot_longer(
    cols = starts_with("prob_positive_split"),
    names_to = "split",
    values_to = "prob_positive"
  )


ggplot(df_one_patient_long, aes(x = File, y = prob_positive, color = split)) +
  geom_point(
    position = position_jitter(width = 0.15, height = 0),
    size = 1.4,
    alpha = 0.7
  ) +
  geom_point(
    data = distinct(df_one_patient_long, File, avg_prob_positive),
    aes(x = File, y = avg_prob_positive),
    inherit.aes = FALSE,
    color = "black",
    size = 1.8
  ) +
  theme_bw() +
  scale_y_continuous(limits = c(0, 1)) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  labs(
    x = "Files ordered by increasing average probability",
    y = "prob_positive",
    color = "Split",
    title = paste("Five split probabilities per cell:", patient_to_plot)
  )


#correct plot
f <- ggplot(df_one_patient, aes(x = cell_index, y = avg_prob_positive)) +
  geom_ribbon(
    aes(
      ymin = avg_prob_positive - sd_prob_positive,
      ymax = avg_prob_positive + sd_prob_positive
    ),
    fill = "#F4A582",
    color = NA,
    alpha = 0.5
  ) +
  geom_point(
    color = "black",
    size = 1
  ) +
  theme_bw() +
  labs(
    x = "Single cell",
    y = "Average AML probability",
    title = paste("Cell-level probability of being AML:", patient_to_plot)
  ) +
  theme(
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 14),
    legend.background = element_rect(
      fill = "white",
      color = "grey70",
      linewidth = 0.5
    ),
    plot.margin = margin(10, 10, 10, 10)
  )
f
ggsave("figures/MDS08_distribution_probabilities_ribbon.pdf", f, width = 5, height = 3, dpi = 300)


# x and y limits and line inside. 
ggplot(
  plot_df,
  aes(x = cell_index, y = avg_prob_positive)
) +
  geom_ribbon(
    aes(
      ymin = pmax(avg_prob_positive - sd_prob_positive, 0),
      ymax = pmin(avg_prob_positive + sd_prob_positive, 1)
    ),
    fill = "#F4A582",
    alpha = 0.25,
    color = NA
  ) +
  geom_line(
    linewidth = 0.8,
    color = "black"
  ) +
  facet_wrap(
    ~ patient,
    ncol = 1,
    scales = "free_x"
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  scale_x_continuous(
    expand = expansion(mult = c(0.01, 0.02))
  ) +
  labs(
    x = "Single cell, ordered by AML probability",
    y = "Average AML probability"
  ) +
  theme_bw(base_size = 11) +
  theme(
    strip.background = element_rect(
      fill = "grey92",
      color = NA
    ),
    strip.text = element_text(
      face = "bold",
      size = 10
    ),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(
      linewidth = 0.3,
      color = "grey90"
    ),
    axis.title = element_text(
      face = "bold",
      size = 11
    ),
    axis.text = element_text(size = 9),
    panel.spacing = unit(0.35, "cm"),
    plot.margin = margin(5, 8, 5, 5)
  )



## plot distribution of all patients together.

df_all_patients <- df %>%
  filter(
    !is.na(patient),
    !is.na(avg_prob_positive),
    !is.na(sd_prob_positive)
  ) %>%
  group_by(patient) %>%
  arrange(avg_prob_positive, .by_group = TRUE) %>%
  mutate(cell_index = row_number()) %>%
  ungroup()

f_facet <- ggplot(
  df_all_patients,
  aes(
    x = cell_index,
    y = avg_prob_positive
  )
) +
  geom_ribbon(
    aes(
      ymin = pmax(avg_prob_positive - sd_prob_positive, 0),
      ymax = pmin(avg_prob_positive + sd_prob_positive, 1)
    ),
    fill = "#F4A582",
    color = NA,
    alpha = 0.5
  ) +
  geom_point(
    color = "black",
    size = 1
  ) +
  facet_wrap(
    ~ patient,
    scales = "free_x",
    ncol = 4
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.25)
  ) +
  theme_bw() +
  labs(
    x = "Single-cell rank within patient",
    y = "Average AML probability",
    title = "Cell-level probability of being AML by patient"
  ) +
  theme(
    strip.background = element_rect(
      fill = "grey90",
      color = NA
    ),
    strip.text = element_text(
      face = "bold",
      size = 11
    ),
    axis.title.x = element_text(
      size = 18,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 18,
      face = "bold"
    ),
    panel.grid.major.y = element_line(
      color = "grey90",
      linewidth = 0.3
    ),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  )

f_facet
ggsave("figures/MDS_distribution_singlecell.pdf", f_facet, width = 7, height = 6.5, dpi = 300)

