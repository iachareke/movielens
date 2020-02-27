################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

#Download the data
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

set.seed(1)

#Test set will be 10% of the edx dataset
#algorithms will be tested on the edx_test set. 
#Only final model will be validated on validation set
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-edx_test_index,]
temp <- edx[edx_test_index,]

# Make sure userId and movieId in edx_test set are also in edx_train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(removed, temp, edx_test_index)

#view data structure
head(edx_train)

#Dimension of dataset
dim(edx_train)

#Investigate distribution of ratings

edx_train %>% group_by(rating) %>% summarize(n= n()) %>%
  ggplot(aes(rating, n)) +
  geom_bar(stat = "identity") +
  ggtitle("Rating distribution")

#investigate top 5 movies by rating, last 5 by rating, and top 5 by number of ratings

ext_movies_1 <- edx_train %>% group_by(title) %>% summarize(rating=mean(rating), n = n()) %>%
  top_n(n= 5, wt= rating)
ext_movies_2 <- edx_train %>% group_by(title) %>% summarize(rating=mean(rating), n = n()) %>%
  top_n(n= -5, wt= rating)
ext_movies_3 <- edx_train %>% group_by(title) %>% summarize(rating=mean(rating), n = n()) %>%
  top_n(n= 5, wt= n)
rbind(ext_movies_1,ext_movies_2, ext_movies_3) %>% 
  ggplot(aes(title, rating)) +
  geom_point(aes(size=n)) +
  coord_flip() +
  ggtitle("Top 5, worst 5 and 5 most rated") +
  ylim(0,6)

# visualize the distribution of ratings per movie
edx_train %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color= "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

#Visualize the user rating patterns per user
edx_train %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Average rating") +
  ylab("Number of users") +
  ggtitle("User rating patterns") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

#visualize the number of rating s given by each user
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

#Visualize the variation of ratings per Genre combination.
edx_train %>% group_by(genres) %>% summarize(rating= mean(rating), n= n()) %>%
  filter(n > 40000) %>%
  mutate(genres = reorder(genres, rating)) %>%
  ggplot(aes(genres, rating)) +
  geom_point() +
  ylim(c(2.5,4)) +
  coord_flip()


#visualize the time effect on ratings
edx_train %>% mutate(date = as_datetime(timestamp)) %>%
  mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth()

#naive mean Model
#calculate mean of all ratings
mu <- mean(edx_train$rating)
mu
#predict mean as rating for all movies
naive_rmse <- RMSE(mu, edx_test$rating)
naive_rmse

#create table to store all results
rmse_results <- data_frame(method = "Average", RMSE = naive_rmse)
rmse_results

#investigate the variation of average rating per movie
edx_train %>% group_by(movieId) %>% summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) + geom_histogram()

#there is a clear variation by movie.
movie_avg <- edx_train %>% group_by(movieId) %>%summarize(b_m = mean(rating - mu))

movie_avg %>% ggplot(aes(b_m)) + geom_histogram()

#Movie bias model
# predict movie rating as mu + b_m
predict_b_m <- mu + edx_test %>% 
  left_join(movie_avg, by='movieId') %>% .$b_m

rmse_b_m <- RMSE(predict_b_m, edx_test$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effect", RMSE = rmse_b_m))
rmse_results %>% knitr::kable

#investigate user effect om ratings
edx_train %>% group_by(userId) %>% summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) + geom_histogram()


#it appears there is a vairation by user. incorporate user bias into model 
user_avg <- edx_train %>% left_join(movie_avg, by='movieId') %>% group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))

user_avg %>% ggplot(aes(b_u)) + geom_histogram()

#movie and user bias model
#predict rating as mu + b_m + b_u
predict_b_u <- edx_test %>% left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  mutate(pred = mu + b_m + b_u) %>% .$pred

rmse_b_u <- RMSE(predict_b_u, edx_test$rating)

rmse_results <- bind_rows(rmse_results, data_frame(method = "User Effect", RMSE = rmse_b_u))
rmse_results %>% knitr::kable()

#Reguliarize user and movie effect 

lambdas <- seq(0,10,0.2)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx_test$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized movie and user effect model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()
lambda

#Add genre effect to regularized movie and user model
l <- lambda
  
  b_i <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
 
  b_g <- edx_train %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - b_i - b_u - mu))
    
    
  predicted_ratings <- 
    edx_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  
rmse_b_g <- RMSE(predicted_ratings, edx_test$rating)
rmse_results<- bind_rows(rmse_results,
                          data_frame(method="Regularized movie, user and Genre effect model",  
                                     RMSE = rmse_b_g))
rmse_results %>% knitr::kable()

# Calculate RMSE on Validation set
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)


rmse_v_g <- RMSE(predicted_ratings, validation$rating)
rmse_results<- bind_rows(rmse_results,
                         data_frame(method="Regularized movie, user and Genre effect model on Validation",  
                                    RMSE = rmse_v_g))
rmse_results %>% knitr::kable()

