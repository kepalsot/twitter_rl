# kepler ps 
# sample script for variable creation 

# libraries 
library(readr)
library(dplyr)
library(tidyr)

# data 
df <- read_csv("path_to_scored_api_data")

# expand user days 
df <- df %>%
  group_by(group, user, user_day) %>%
  complete(values = full_seq(values, 1)) %>%  # include days without posts (to be able to create day 2 sentiment properly)

# create variables of interest at the user day-level 
df <- df %>%
  group_by(group, user, user_day)
  mutate(
    log_avg_likes_d1 = log(mean(favorite_count) + 1),  # set minimum favorites to 1 to avoid NANs
    log_max_liked_likes = log(max(favorite_count + 1)), # the amount of likes of the max-liked post
    avg_sentiment_d1 = mean(polarity), 
    max_liked_sent_d1 = mean(polarity[(favorite_count == max(favorite_count))], na.rm = TRUE)) # sentiment of max-liked post 

# get next-day avg sentiment in same row 
df <- df %>%
  group_by(group, user) %>%
  mutate(avg_sentiment_d2 = lead(avg_sentiment_d1)) # omits one row per user! 

# labeling viral posts
viral_only <- post_level %>% 
  # for each user day, find the maximum likes received across all posts 
  group_by(group, user, user_day) %>%
  filter(favorite_count == max(favorite_count)) %>%

  # for each user, find the 25th and 75th pctiles for max-liked posts 
  group_by(group, user) %>%
  mutate(pctile_75 = quantile(favorite_count, .75, na.rm = TRUE), 
         pctile_25 = quantile(favorite_count, .25, na.rm = TRUE), 
         neutral = polarity <= -.2 & polarity <=.2) %>%
  
  # filter users with low variance in likes
  filter(pctile_75 != pctile_25) %>%  

  # keep only user days where the max-liked post exceeds the 75th %tile or is below the 25th %tile for likes (of max-liked posts)
  # remove neutral posts 
  filter(favorite_count > pctile_75 | favorite_count <= pctile_25, !neutral) %>%
  
  # for regression purposes, create a logical vector column for membership above the 75th pctile (TRUE) or below the 25th pctile (FALSE)
  # logical vector column of whether the post was positive (TRUE) or negative (FALSE)
  mutate(high_liked = favorite_count > pctile_75)


dim(viral_only)

# join viral data label to main df 
df <- df %>%
  left_join(viral_only, by = c("user", "created_at"))

df.write_csv('ready_for_analyses_path')

