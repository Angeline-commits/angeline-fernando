#' ---
#' title: "Cluster Analysis -Shopping Data"
#' ---

#'Reset environment before starting
set.seed(NULL)
rm(list = ls(all.names = TRUE))

#'Packages needed 
#factoextra : to determine the optimal number clusters data visualization.
#fpc: cluster validation with silhouette width
library(factoextra)
library(fpc)

#'Load the data
# Read the contents of the csv file into a variable called shopping_original
# "header=TRUE" implies that header is present in the data file

shopping_original <- read.csv("./data/Shopping_Data.csv", header = TRUE)

#'Examine the dataset

#str shows the structure of the object :variable type
str(shopping_original)
#summary produces descriptive stats for the variables in the data
summary(shopping_original)

#'Preprocessing
#We do not want to cluster customer id -caseNo in the first column
#Remove this and create a new dataframe to work with

shopping.df<- shopping_original[ , -c(1)]  # Drop caseNo columns.


#'Set the seed 
#Refer : http://rfunction.com/archives/62 K-means algorithm uses random number
#simulations to generate solutions.
# When we use any algorithm that uses random #number to simulate 
# we have to set the seed in r environment to get reproducible results
set.seed(96743)

#'Number of clusters
#To identify number of clusters apriori
# Use wss and silhouette metrics


#Plots Total within-clusters sum of squares vs. Number of clusters
fviz_nbclust(shopping.df, kmeans, method = "wss")

#'The plot shows the elbow bend at **3 **
#Beyond this bend there is no value obtained in terms of wss
#Plots silhouette

fviz_nbclust(shopping.df, kmeans, method = "silhouette")

#'The plot shows the peak at **3 ** again


#Both the methods show that the data contains 3 clusters
#'Run k-means with 3 clusters
shopping.seg <- kmeans(shopping.df,centers = 3)
shopping.seg

#'The results show the following
# 1. The number of clusters and the number of rows in each cluster
# 2. The means of the variables in each cluster
# 3. Clustering vector: A vector of integers that shows the cluster each observation belongs to

# Cluster size
shopping.seg$size

# Cluster means
shopping.seg$centers

#'Visualise the clusters

fviz_cluster(shopping.seg, data = shopping.df)

#'Tabulate the clusters and Profile
# K-means clustering with 3 clusters of sizes 8, 6, 6 90, 16
# V1	Shopping is fun
# V2	Shopping is bad for your budget
# V3	I combine shopping with eating out
# V4	I try to get the best buys when shopping
# V5	I don't care about shopping
# V6	You can save a lot of money by comparing prices
# (6-Highly Agree and 1 - Highly Disagree).

#     V1       V2       V3      V4    V5       V6
# 1 3.500000 5.833333 3.333333 6.000 3.500 6.000000
# 2 1.666667 3.000000 1.833333 3.500 5.500 3.333333
# 3 5.750000 3.625000 6.000000 3.125 1.875 3.875000

#Each row represents a cluster
#Get the max value loaded into each cluster
# Row 1 - Price conscious shoppers-V4 and V6 have the highest loading i.e 6.0 
# followed by V2
# Row 2 - Apathetic Shoppers-V5 is the highest, followed by V4
# Row 3 - Fun loving-eating out-V3 and V1

mycluster_profile<-data.frame(
  cbind(shopping_original,"cluster_number"=shopping.seg$cluster))
# Add the names we gave for the clusters

mycluster_profile$cluster_name[mycluster_profile$cluster_number==1] <- "Price Conscious"
mycluster_profile$cluster_name[mycluster_profile$cluster_number==2] <- "Apathetic"
mycluster_profile$cluster_name[mycluster_profile$cluster_number==3] <- "Fun-Loving"

#Write the file to excel for further analysis
write.csv(mycluster_profile,file = "cluster_profiles.csv")



