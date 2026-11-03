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



# Folder with the input tables (embeddings, dataset summaries, metrics summaries); set AINU_R_DATA or run from it.
setwd(Sys.getenv("AINU_R_DATA", "."))
dir.create("figures", showWarnings = FALSE)
# 1) Read data (tab-separated)
df_original <- read_tsv("MDS_splits_metrics_summary_CORRECT.txt", show_col_types = FALSE)
df_original <- read_tsv("MDS_splits_metrics_summary_test_sample2.txt", show_col_types = FALSE)
df_original <- read_tsv("MDS_splits_metrics_summary_test_sample4.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_MDS.txt", show_col_types = FALSE)


df_original <- read_tsv("splits_metrics_summary_onepatout_test_complete.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_AML.txt", show_col_types = FALSE)

# Rename the patients with the code
df <- df_original %>%
  left_join(
    data %>% select(patientID, patient),
    by = c("Patient" = "patientID")
  )


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    .groups = "drop"
  )


# Define institute colors manually (optional but cleaner)
npg_colors <- pal_npg("nrc")(4)
npg_colors
custom_colors <- c(
  "Hospital1" = npg_colors[2],
  "Hospital2" = npg_colors[1],
  "Hospital3" = npg_colors[4],
  "Commercial" = npg_colors[3]
)

# Create colored axis labels
# DO NOT RUN 
summary_df <- summary_df %>%
  mutate(
    Patient_label = paste0(
      "<span style='color:",
      inst_colors[Institute],
      "'>", patient, "</span>"
    )
  )

# add follow up 
summary_df <- summary_df %>%
  mutate(follow_up = case_when(
    patient %in% c("MDS02", "MDS03") ~ "Improved",
    patient %in% c("MDS09","MDS11", "MDS12", 
                   "CMML01","CMML03", "MDS14", "CMML02", "CCUS01" ) ~ "No_follow_up",
    patient %in% c("MDS-AML16", "MDS-AML17", "MDS07") ~ "Worsen",
    patient %in% c("MDS06", "MDS08", "MDS01") ~ "Unchanged",
    TRUE ~ NA_character_
  ))

custom_colors = c(
  "Unchanged" = "#4DBBD5FF",
  "Improved" = "#00A087FF",
  "Worsen" = "#E64B35FF",
  "No_follow_up" = "black"
)


# --------------------
# Plot for publication 
#----------------------
# plot institute
p <- ggplot(summary_df,
       aes(x = reorder(patient, mean_accuracy),
           y = mean_accuracy,
           shape = Origin,
           color = Institute)) +   # <-- This creates legend
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  scale_color_manual(values = custom_colors) +
  coord_flip() +
  ylim(0.00,1.00) +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy for the AML class",
    color = "Institute",
    shape = "Sample type"
  ) +
  geom_hline(yintercept=(0.70),color='red') +
  geom_hline(yintercept=(0.60),color='green') +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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


#plot follow up
p <- ggplot(summary_df,
                         aes(x = reorder(patient, mean_accuracy),
                             y = mean_accuracy,
                             shape = Origin,
                             color = follow_up)) +   # <-- This creates legend
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  scale_color_manual(values = custom_colors) +
  coord_flip() +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy",
    color = "4-10 moths follow up",
    shape = "Sample type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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
ggsave("figures/onepatout_accuracy.pdf", p, width = 7, height = 8, dpi = 300)
ggsave("figures/onepatout_accuracy_lines.pdf", p, width = 7, height = 8, dpi = 300)
ggsave("figures/MDS_onepatout_followup.pdf", p, width = 7, height = 6, dpi = 300)
ggsave("figures/MDS_test_on_model4.pdf", p, width = 7, height = 6, dpi = 300)




#plot two mds test at the same time 

df_original2 <- read_tsv("MDS_splits_metrics_summary_test_sample2.txt", show_col_types = FALSE)
df_original4 <- read_tsv("MDS_splits_metrics_summary_test_sample4.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_MDS.txt", show_col_types = FALSE)


# Rename the patients with the code
df2 <- df_original2 %>%
  left_join(
    data %>% select(patientID, patient, diagnosis),
    by = c("Patient" = "patientID")
  )
df4 <- df_original4 %>%
  left_join(
    data %>% select(patientID, patient, diagnosis),
    by = c("Patient" = "patientID")
  )


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df2 <- df2 %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    diagnosis     = first(diagnosis),
    .groups = "drop"
  ) %>%
  mutate(Model = "Model 2")

summary_df4 <- df4 %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    
    .groups = "drop"
  ) %>%
  mutate(Model = "Model 4")

summary_combined <- bind_rows(summary_df2, summary_df4) %>%
  mutate(patient = factor(patient))

## STATS ##
# normality test 
df2_model <- df2 %>% mutate(Model = "Model 2")
df4_model <- df4 %>% mutate(Model = "Model 4")

df_combined <- bind_rows(df2_model, df4_model) %>%
  mutate(patient = factor(patient))
df_combined %>%
  group_by(patient) %>%
  summarise(p_value = shapiro.test(Test_Accuracy)$p.value)
df_combined %>%
  group_by(patient) %>%
  summarise(
    t_p = t.test(Test_Accuracy ~ Model)$p.value
  )

pvals_df <- df_combined %>%
  group_by(patient) %>%
  summarise(
    p_value = t.test(Test_Accuracy ~ Model)$p.value,
    .groups = "drop"
  ) %>%
  filter(p_value < 0.05) %>%   # keep only significant tests
  mutate(
    label = paste0("p = ", signif(p_value, 2))
  )
label_df <- summary_combined %>%
  group_by(patient) %>%
  summarise(
    y = max(mean_accuracy + sd_accuracy, na.rm = TRUE) + 0.03,
    .groups = "drop"
  ) %>%
  left_join(pvals_df, by = "patient")


# Plot with facet wrap
p <- ggplot(summary_combined,
            aes(x = patient,
                y = mean_accuracy,
                shape = Origin,
                color = Institute)) +
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  geom_text(data = label_df,
            aes(x = patient, y = y, label = label),
            inherit.aes = FALSE,
            size = 4) +
  scale_color_manual(values = custom_colors) +
  coord_flip(ylim = c(0, 1)) +
  facet_wrap(~Model, ncol = 2) +
  labs(
    x = "Patient",
    y = "Mean Test Accuracy for the AML class",
    color = "Institute",
    shape = "Sample type"
  ) +
  theme_bw(base_size = 14)
p
ggsave("figures/combined_MDS_test_on_model2&4_pvalues.pdf", p, width = 8.5, height = 6, dpi = 300)


# Plot combined in one graph and with threshold lines
p <- ggplot(summary_combined,
            aes(x = patient,
                y = mean_accuracy,
                shape = Origin,
                color = Model)) +
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  scale_color_manual(values = c(
    "Model 2" = "#E64B35FF",
    "Model 4" = "#3C5488FF")) +
  coord_flip(ylim = c(0, 1)) +
  labs(
    x = "Patient",
    y = "Mean Test Accuracy for the AML class",
    color = "Model",
    shape = "Sample type"
  ) +
  geom_hline(yintercept=(0.70),color='red') +
  geom_hline(yintercept=(0.60),color='green') +
  theme_bw(base_size = 14)
p

p_diagnosis <- ggplot(summary_combined,
                 aes(x = 1, y = patient, label = diagnosis)) +
  geom_text(hjust = 0) +
  xlim(1, 1.5) +
  theme_void() +
  theme(
    plot.margin = margin(10, 10, 10, 0)
  )


#p_all <- p + p_diagnosis + plot_layout(widths = c(4, 1))

ggsave("figures/combined_MDS_test_on_model2&4_lines.pdf", p , width = 6, height = 6, dpi = 300)
ggsave("figures/MDS_diagnosis.pdf", p_diagnosis , width = 3, height = 6, dpi = 300)



# ---- Plot 2: Scatterplot Images vs Mean Accuracy ----
d <- ggplot(summary_df,
             aes(x = Images,
                 y = mean_accuracy,
                 color = Institute,
                 shape = Origin)) +
  geom_point(size = 5) +
  scale_color_manual(values = custom_colors) +
  #geom_smooth(method = "lm", se = TRUE) +
  labs(
   # title = "Mean Accuracy vs Number of Images",
    x = "Number of Images",
    y = "Mean Test Accuracy",
    color = "Institute",
    shape = "Sample type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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
d
ggsave("figures/scatterplot.pdf", d, width = 7, height = 5, dpi = 300)


sp5 <- c(0.7537878787878788,
          0.75,
          0.7613636363636364,
          0.7613636363636364,
          0.6553030303030303)
mean = mean(sp5, na.rm = TRUE)
sd   = sd(sp5, na.rm = TRUE)
mean
sd

# Optional: save
# ggsave("mean_accuracy_per_patient.png", p1, width = 10, height = 8, dpi = 300)
# ggsave("accuracy_vs_images.png", p2, width = 8, height = 6, dpi = 300)


##-----------------------------------------
##patient retrain with 10 and 20 images
##-----------------------------------------
data <- read_tsv("retrain_AML17_metrics_summary.txt", show_col_types = FALSE)
data <- read_tsv("retrain_AML06_metrics_summary.txt", show_col_types = FALSE)
data <- read_tsv("retrain_AML14_metrics_summary.txt", show_col_types = FALSE)
data <- read_tsv("retrain_AML23_metrics_summary.txt", show_col_types = FALSE)
data <- read_tsv("retrain_AML27_metrics_summary.txt", show_col_types = FALSE)

# 2) Summarise mean + sd per patient (and keep metadata)
summary_data <- data %>%
  group_by(Images) %>%
  summarise(
    mean_accuracy = mean(Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Accuracy, na.rm = TRUE),
    .groups = "drop"
  )

p <- ggplot(summary_data, aes(x = Images, y = mean_accuracy))+
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 3) +
  ylim(0,1) +
  labs(
    title = "AML27  retraining",
    x = "n of images in retraing",
    y = "Mean Test Accuracy") +
  theme_minimal(base_size = 14)
p
  
ggsave("retrain_AML17.png", p, width = 4, height = 6, dpi = 300)
ggsave("retrain_AML06.png", p, width = 4, height = 6, dpi = 300)
ggsave("retrain_AML14.png", p, width = 4, height = 6, dpi = 300)
ggsave("retrain_AML23.png", p, width = 4, height = 6, dpi = 300)
ggsave("retrain_AML27.png", p, width = 4, height = 6, dpi = 300)

library(dplyr)
library(ggplot2)
library(patchwork)  # For combined plot

#-----------------------------------------------
# Read all 5 datasets and add Patient column
#-----------------------------------------------
AML17 <- read_tsv("retrain_AML17_metrics_summary.txt", show_col_types = FALSE)
#AML06 <- read_tsv("retrain_AML06_metrics_summary.txt", show_col_types = FALSE)
#AML14 <- read_tsv("retrain_AML14_metrics_summary.txt", show_col_types = FALSE)
#AML23 <- read_tsv("retrain_AML23_metrics_summary.txt", show_col_types = FALSE)
AML27 <- read_tsv("retrain_AML27_metrics_summary.txt", show_col_types = FALSE)

AML06 <- read_tsv("retrain_AML06_seed43_metrics_summary.txt", show_col_types = FALSE)
AML14 <- read_tsv("retrain_AML14_seed43_metrics_summary.txt", show_col_types = FALSE)
AML23 <- read_tsv("retrain_AML23_seed43_metrics_summary.txt", show_col_types = FALSE)

AML17$patient <- "AML17"
AML06$patient <- "AML06"
AML14$patient <- "AML14"
AML23$patient <- "AML23"
AML27$patient <- "AML27"

df_all <- bind_rows(AML17, AML06, AML14, AML23, AML27)
#df_all <- bind_rows(AML17, AML06, AML27)


df_summary <- df_all %>%
  group_by(patient, Images) %>%
  summarise(
    mean_accuracy = mean(Accuracy, na.rm = TRUE),
    sd_accuracy = sd(Accuracy, na.rm = TRUE),
    .groups = "drop"
  )

## STATS ##
# normality test 
df_all %>%
  group_by(patient, Images) %>%
  summarise(p_value = shapiro.test(Accuracy)$p.value)
#normally distributed
anova <- df_all %>%
  group_by(patient) %>%
  anova_test(Accuracy ~ Images)
pairwise_results <- df_all %>%
  group_by(patient) %>%
  tukey_hsd(Accuracy ~ Images) %>%
  add_y_position(formula = Accuracy ~ Images, fun = "mean_sd", step.increase = 0.2)

anova
pairwise_results 

my_palette <- rev(magma(8))[c(-1, -8)]
my_palette
# [1] "#FEBA80FF" "#F8765CFF" "#D3436EFF" "#982D80FF" "#5F187FFF" "#231151FF"

p <- ggplot(df_summary, aes(x = Images, y = mean_accuracy, fill = Images)) +
  geom_col(
    width = 0.5,
    linewidth = 0.5
  ) +
  geom_errorbar(
    aes(
      ymin = mean_accuracy - sd_accuracy,
      ymax = mean_accuracy + sd_accuracy
    ),
    width = 0.3,
    linewidth = 1,
    color = "darkgrey"
  ) +
  scale_fill_manual(values = c( 
    "0img" = "#FEBA80FF", 
    "10img" = "#D3436EFF", 
    "20img" = "#5F187FFF")) +
  stat_pvalue_manual(
    pairwise_results,
    label = "p.adj.signif",
    xmin = "group1",
    xmax = "group2",
    y.position = "y.position",
    tip.length = 0.01,
    hide.ns = TRUE,
    inherit.aes = FALSE,
    bracket.size = 0.8,  # thickness of bracket lines
    size = 6             # size of stars/text
  ) +
  facet_wrap(~ patient) +
  ylim(0, 1.2) +
  labs(
    x = "Images in training",
    y = "Mean Accuracy",
    fill = "Images"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.background = element_rect(fill = "white", color = "grey70", linewidth = 0.5),
    plot.margin = margin(10, 10, 10, 10),
    legend.position = "bottom"
  )

p


ggsave("figures/retrain_complete_2.pdf", p, width = 5, height = 6.5, dpi = 300)
ggsave("figures/retrain_columns_pvalue.pdf", p, width = 5.5, height = 6.5, dpi = 300)
ggsave("figures/retrain_complete_3.pdf", p, width = 4, height = 5, dpi = 300)
ggsave("figures/retrain_complete_facet_wrap.pdf", p, width = 4, height = 5, dpi = 300)



##--------- MRD test on sample 2, 4 and 7 model --------


df_original <- read_tsv("MRD_test_sample4_splits_metrics_summary.txt", show_col_types = FALSE)
df_original <- read_tsv("MRD_test_sample7_splits_metrics_summary.txt", show_col_types = FALSE)
df_original <- read_tsv("MRD_test_sample2_splits_metrics_summary.txt", show_col_types = FALSE)


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df_original %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    .groups = "drop"
  ) 
#%>% 
  #mutate(
   # highlight = patient == "AML32-MRD-W",
   # shape_flag = ifelse(highlight, "square", "circle"),
   # Patient_label = patient
   # )

# add follow up 
summary_df <- summary_df %>%
  mutate(follow_up = case_when(
    patient %in% c("MRD01","MRD02-1","MRD02-2","MRD02-3","MRD03","MRD04",
                   "MRD06","MRD07","MRD11","MRD12") ~ "Unchanged",
    patient %in% c("MRD05","MRD09") ~ "No_follow_up",
    patient == "MRD08" ~ "Non-measurable_MRD",
    patient %in% c("AML32-MRD", "AML32-MRD-W") ~ "Relapse",
    patient %in% c("MRD10-1","MRD10-2","MRD10-3","CMML04") ~ "Improved",
    TRUE ~ NA_character_
  ))


# Plot
a <- ggplot(summary_df,
       aes(x = reorder(patient, mean_accuracy),
           y = mean_accuracy
           #color = hospital,
           #shape = highlight
           )) +   
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 5) +
  #scale_color_manual(
  #  values = c(
  #    "Unchanged" = "#4DBBD5FF",
  #    "Improved" = "#00A087FF",
  #    "Relapse" = "#E64B35FF",
  #    "No_follow_up" = "black",
  #    "Non-measurable_MRD" = "#3C5488FF"
  #  )) +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy"
    #color = "4-10 months follow_up"
  ) +
  guides(
    shape = "none"
  )+
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
    strip.background = element_rect(fill = "grey90", color = NA),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(size = 14),
    axis.title.x = element_text(size = 18, face = "bold"),
    axis.title.y = element_text(size = 18, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    legend.background = element_rect(fill = "white", color = "grey70", linewidth = 0.5),
    plot.margin = margin(10, 10, 10, 10)
  )

a
ggsave("figures/mrd_test_sample4_v2.pdf", a, width = 5.5, height = 7, dpi = 300)


#plot two mrd test at the same time 

df_original4 <- read_tsv("MRD_test_sample4_splits_metrics_summary.txt", show_col_types = FALSE)
df_original2 <- read_tsv("MRD_test_sample2_splits_metrics_summary.txt", show_col_types = FALSE)


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df2 <- df_original2 %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Model = "Model 2")

summary_df4 <- df_original4 %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(Model = "Model 4")

summary_combined <- bind_rows(summary_df2, summary_df4) %>%
  mutate(patient = factor(patient))

summary_combined <- summary_combined %>%
  mutate(follow_up = case_when(
    patient %in% c("MRD01","MRD02-1","MRD02-2","MRD02-3","MRD03","MRD04",
                   "MRD06","MRD07","MRD11","MRD12") ~ "Unchanged",
    patient %in% c("MRD05","MRD09") ~ "No_follow_up",
    patient == "MRD08" ~ "Non-measurable_MRD",
    patient %in% c("AML32-MRD") ~ "Relapse",
    patient %in% c("MRD10-1","MRD10-2","MRD10-3","CMML04") ~ "Improved",
    TRUE ~ NA_character_
  ))

p <- ggplot(summary_combined,
            aes(x = patient,
                y = mean_accuracy,
                color = follow_up)) +
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  geom_text(data = label_df,
            aes(x = patient, y = y, label = label),
            inherit.aes = FALSE,
            size = 4) +
  scale_color_manual(values = c(
    "Unchanged" = "#4DBBD5FF",
    "Improved" = "#00A087FF",
    "Relapse" = "#E64B35FF",
    "No_follow_up" = "black",
    "Non-measurable_MRD" = "#3C5488FF")) +
  coord_flip(ylim = c(0, 1)) +
  facet_wrap(~Model, ncol = 2) +
  labs(
    x = "Patient",
    y = "Mean Test Accuracy for the AML class",
    color = "follow up",
  ) +
  theme_bw(base_size = 14)

p <- ggplot(summary_combined,
            aes(x = patient,
                y = mean_accuracy,
                color = Model)) +
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 4) +
  scale_color_manual(values = c(
    "Model 2" = "#E64B35FF",
    "Model 4" = "#3C5488FF")) +
  coord_flip(ylim = c(0, 1)) +
  labs(
    x = "Patient",
    y = "Mean Test Accuracy for the AML class",
    color = "Model",
  ) +
  geom_hline(yintercept=(0.70),color='red') +
  geom_hline(yintercept=(0.60),color='green') +
  theme_bw(base_size = 14)
p
ggsave("figures/combined_MRD_test_on_model2&4_lines.pdf", p, width = 6, height = 6, dpi = 300)

data <- read_tsv("dataset_summary_MRD.txt", show_col_types = FALSE)
p_diagnosis <- ggplot(data,
                      aes(x = 1, y = patient, label = diagnosis)) +
  geom_text(hjust = 0) +
  xlim(1, 1.5) +
  theme_void() +
  theme(
    plot.margin = margin(10, 10, 10, 0)
  )
p_diagnosis
ggsave("figures/MRD_diagnosis.pdf", p_diagnosis, width = 3, height = 3, dpi = 300)





## STATS ##
# normality test 
df2_model <- df_original2 %>% mutate(Model = "Model 2")
df4_model <- df_original4 %>% mutate(Model = "Model 4")

df_combined <- bind_rows(df2_model, df4_model) %>%
  mutate(patient = factor(patient))
df_combined %>%
  group_by(patient) %>%
  summarise(p_value = shapiro.test(Test_Accuracy)$p.value)
df_combined %>%
  group_by(patient) %>%
  summarise(
    t_p = t.test(Test_Accuracy ~ Model)$p.value
  )

pvals_df <- df_combined %>%
  group_by(patient) %>%
  summarise(
    p_value = t.test(Test_Accuracy ~ Model)$p.value,
    .groups = "drop"
  ) %>%
  filter(p_value < 0.05) %>%   # keep only significant tests
  mutate(
    label = paste0("p = ", signif(p_value, 2))
  )
label_df <- summary_combined %>%
  group_by(patient) %>%
  summarise(
    y = max(mean_accuracy + sd_accuracy, na.rm = TRUE) + 0.03,
    .groups = "drop"
  ) %>%
  left_join(pvals_df, by = "patient")
ggsave("figures/combined_MRD_test_on_model2&4_pvalues.pdf", p, width = 10, height = 6, dpi = 300)







library("scales")
show_col(pal_npg("nrc")(10))
show_col(pal_npg("nrc", alpha = 0.6)(10))


#------------------------------------------------------------sample 3 mrd 

df_original <- read_tsv("MRD_test_sample3_splits_metrics_summary.txt", show_col_types = FALSE)



# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df_original %>%
  group_by(patient) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    .groups = "drop"
  )

summary_df$patient <- factor(summary_df$patient, levels = c("MRD03","MRD07","MRD11","MRD02-1","MRD01","MRD08","MRD10-2",
                                                            "MRD10-1","MRD12","MRD02-2","MRD06","AML32-MRD","MRD02-3",
                                                            "MRD05","MRD10-3","MRD09","MRD04"))



# Plot
a <- ggplot(summary_df,
            aes(x = patient,
                y = mean_accuracy)) +   
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40") +
  geom_point(size = 3) +
  #scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "black"), guide = "none") +
  scale_y_continuous(limits = c(0, 1)) +
  coord_flip() +
  labs(
    title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = NULL,
    y = "Mean Test Accuracy"
  ) +
  theme_bw(base_size = 14) +
  theme(axis.text.y = ggtext::element_markdown(size = 14)
  )
a

ggsave("mrd_test_sample3.png", a, width = 7, height = 7, dpi = 300)


#----------------------
# retrain patients aml 21 24 28 
#----------------------

df_original <- read_tsv("AML21-24-28_retraining.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_AML.txt", show_col_types = FALSE)



# Rename the patients with the code
df <- df_original %>%
  left_join(
    data %>% select(patientID, patient),
    by = c("Patient" = "patientID")
  )

#df_list <- split(df, df$patient)


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df %>%
  group_by(patient, training) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    .groups = "drop"
  )
summary_df$training <- recode(summary_df$training,
                              "PB_trained_on_BM" = "BM (same patient)",
                              "PB_trained_on_nothing" = "No patient-specific training",
                              "BM_trained_on_PB" = "PB (same patient)",
                              "BM_trained_on_nothing" = "No patient-specific training"
)


# Define institute colors manually (optional but cleaner)
npg_colors <- pal_npg("nrc")(4)
npg_colors
custom_colors <- c(
  "PB_trained_on_BM" = npg_colors[3],
  "no cells of that patient" = "#91D1C2FF",
  "BM_trained_on_PB" = npg_colors[4],
  "no cells of that patient" = "#8491B4FF"
)

custom_colors <- c(
  "BM (same patient)" = npg_colors[3],
  "PB (same patient)" = npg_colors[4],
  "No patient-specific training" = npg_colors[1]
)

# Plot for publication 

p <- ggplot(summary_df,
            aes(x = patient,
                y = mean_accuracy,
                #shape = Origin,
                color = training,
                group = training)) +   
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40",
                position = position_dodge(width = 0.5)) +
  geom_point(size = 4, 
             position = position_dodge(width = 0.5)) +
  scale_color_manual(values = custom_colors) +
  coord_flip() +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy",
    color = "Model trained on:",
    #shape = "Sample type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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
ggsave("figures/reatraining_21_24_28v2.pdf", p, width = 7, height = 4, dpi = 300)





#----------------------
# retest patients MDS_AML16 and MDS_AML17 wihtout AML in training 
#----------------------

df_original <- read_tsv("MDS_comparison_MDS-AML16_17.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_.txt", show_col_types = FALSE)


# Rename the patients with the code
df <- df_original %>%
  left_join(
    data %>% select(patientID, patient),
    by = c("Patient" = "patientID")
  )

#df_list <- split(df, df$patient)


# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df %>%
  group_by(patient, training) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    .groups = "drop"
  )

# Define institute colors manually (optional but cleaner)
npg_colors <- pal_npg("nrc")(4)
npg_colors

custom_colors <- c(
  "with_AML" = npg_colors[3],
  "without_AML" = npg_colors[4]
)

# Plot for publication 

p <- ggplot(summary_df,
            aes(x = patient,
                y = mean_accuracy,
                #shape = Origin,
                color = training,
                group = training)) +   
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40",
                position = position_dodge(width = 0.5)) +
  geom_point(size = 4, 
             position = position_dodge(width = 0.5)) +
  scale_color_manual(values = custom_colors) +
  coord_flip() +
  ylim(0.0, 1.0) +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy",
    color = "Model trained on:",
    #shape = "Sample type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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
ggsave("figures/reatraining_MDS-AML16_17.pdf", p, width = 7, height = 3, dpi = 300)

## STATS ##
# normality test 
df %>%
  group_by(patient, training) %>%
  summarise(p_value = shapiro.test(Test_Accuracy)$p.value)
#normally distributed
aov(Test_Accuracy ~ training, data = df)
df %>%
  group_by(patient) %>%
  summarise(
    p_value = t.test(Test_Accuracy ~ training)$p.value
  )





#-----------------------------
# comparison AML32-MRD for training on sample4 and 7 with and without aml 
# ------------------------------

df <- read_tsv("MRD_comparison_AML32_MRD_samples4&7.txt", show_col_types = FALSE)

# 2) Summarise mean + sd per patient (and keep metadata)
summary_df <- df %>%
  group_by(sample, training) %>%
  summarise(
    mean_accuracy = mean(Test_Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Test_Accuracy, na.rm = TRUE),
    Origin        = first(Origin),
    Institute     = first(Institute),
    Images        = first(Images),
    .groups = "drop"
  )

# Define institute colors manually (optional but cleaner)
npg_colors <- pal_npg("nrc")(4)
npg_colors

custom_colors <- c(
  "with_AML" = npg_colors[1],
  "without_AML" = npg_colors[4]
)

# Plot for publication 

p <- ggplot(summary_df,
            aes(x = sample,
                y = mean_accuracy,
                #shape = Origin,
                color = training,
                group = training)) +   
  geom_errorbar(aes(ymin = mean_accuracy - sd_accuracy,
                    ymax = mean_accuracy + sd_accuracy),
                width = 0.2,
                color = "grey40",
                position = position_dodge(width = 0.5)) +
  geom_point(size = 4, 
             position = position_dodge(width = 0.5)) +
  scale_color_manual(values = custom_colors) +
  coord_flip() +
  ylim(0.0, 1.0) +
  labs(
    #title = "Mean Test Accuracy per Patient (5 splits) ± SD",
    x = "Patient",
    y = "Mean Test Accuracy",
    color = "Model trained on:",
    #shape = "Sample type"
  ) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.y = ggtext::element_markdown(size = 14),
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
ggsave("figures/reatraining_AML32-MRD.pdf", p, width = 7, height = 3, dpi = 300)

## STATS ##
# normality test 
df %>%
  group_by(patient, training) %>%
  summarise(p_value = shapiro.test(Test_Accuracy)$p.value)
#normally distributed
aov(Test_Accuracy ~ training, data = df)
df %>%
  group_by(patient) %>%
  summarise(
    p_value = t.test(Test_Accuracy ~ training)$p.value
  )




###################################
# MDS and MRD tested on 4 classes 
#################################

df_original <- read_tsv("accuracy_summary_MDS_4classes.txt", show_col_types = FALSE)
data <- read_tsv("dataset_summary_MDS.txt", show_col_types = FALSE)

df_original <- read_tsv("summary_MRD_test_4classes.txt", show_col_types = FALSE)
data <- read_tsv("MRD_dataset_summary.txt", show_col_types = FALSE)

df <- df_original %>%
  left_join(
    data %>% select(patientID, patient),
    by = c("Sample" = "patientID")
  )
#make it tidy
df_tidy <- df %>%
  pivot_longer(
    cols = c(AML_BM, AML_PB, Control_BM, Control_PB),
    names_to = "Class",
    values_to = "Accuracy"
  ) %>%
  mutate(
    Split = str_remove(Split, "split") %>% as.integer(),
    Class = factor(Class, levels = c("AML_BM", "AML_PB", "Control_BM", "Control_PB"))
  )


summary_df <- df_tidy %>%
  group_by(patient, Class) %>%
  summarise(
    mean_accuracy = mean(Accuracy, na.rm = TRUE),
    sd_accuracy   = sd(Accuracy, na.rm = TRUE),
    .groups = "drop"
  )


y <- ggplot(summary_df, aes(x = Class, y = patient, fill = mean_accuracy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(mean_accuracy, 2), 
            color = ifelse(mean_accuracy > 0.4, "white", "black")), size = 4) +
  scale_color_identity() +
  scale_fill_gradient(
    low  = "#C6DBEF",
    high = "#08306B",
    limits = c(0, 1),
    name = "Mean accuracy"
  ) +
  scale_x_discrete(position = "top") +
  theme_minimal(base_size = 12) +
  theme(
    axis.title        = element_blank(),
    axis.text.x       = element_text(size = 12, face = "bold"),
    axis.text.y       = element_text(size = 14),
    panel.grid        = element_blank(),
    legend.position   = "bottom",
    legend.key.width  = unit(2, "cm"),
    strip.text = element_text(face = "bold"),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 14),
  )
y
ggsave("figures/mds_test_4classes_heatmap.pdf", y, width = 6, height = 7, dpi = 300)
ggsave("figures/mrd_test_4classes_heatmap.pdf", y, width = 6, height = 7, dpi = 300)


custom_colors <- c(
  "AML_BM" = "#BCE4D8",
  "AML_PB" = "#439FB7",
  "Control_BM" = "#7BBFC9",
  "Control_PB" = "#2C5985")

p <- ggplot(summary_df, aes(x = Class, y = mean_accuracy, fill = Class)) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(ymin = mean_accuracy - sd_accuracy, ymax = mean_accuracy + sd_accuracy),
    width = 0.2
  ) +
  facet_wrap(~ patient) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right", 
    axis.text.y = ggtext::element_markdown(size = 14),
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
    x = "Predicted class",
    y = "Mean accuracy across splits"
  )
p
ggsave("figures/mds_test_4classes.pdf", p, width = 7, height = 7, dpi = 300)
ggsave("figures/mrd_test_4classes.pdf", p, width = 8, height = 7.5, dpi = 300)




       