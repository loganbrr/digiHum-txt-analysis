## RStudio allows you to run each line as you work with Run above.
## See results in bottom right.

## First install language processing packages. Go to tools, Install Packages, then type CoreNLP.

## Install the Packages
# install.packages("syuzhet")
# install.packages("RColorBrewer")
# install.packages("wordcloud")
# install.packages("tm")


## Packages are groups of functions that you can use.
## We are putting the following packages into the Library to use.
library("readr")
library("qdap")
library("syuzhet")
library("ggplot2")
library("dplyr")

## If you don't know what a package is, you can ask. Use ctrl-return or press Run above.
## I have placed these queries in the code, but better practice is to type it directly
## in the console below.

# ??syuzhet
# ??qdap
# ??twitteR

## make VERY sure that the following packages have loaded-- base, datasets, dplyr, ggplot2, graphics, grDevices, methods, plyr, qdap, qdapDictionaries, qdapRegex, qdapTools, RColorBrewer, RCurl, RDSTK, readr, rjson, stats, syuzhet, twitteR, utils

fomc_minutes = read_file("assets/cleaned_fomc.txt")
View(fomc_minutes)

## POLARITY AND SENTIMENT
## in general (for us), sentiment is how postive, negative, or neutral a statement is.
## polarity is the degree of that sentiment
g_scores = get_nrc_sentiment(fomc_minutes)
class(g_scores)
g_scores
g_polarity = g_scores[1,9:10]
g_sentiment = g_scores[1,1:8]

## visualize polarity
class(g_polarity)
g_polarity = data.matrix(g_polarity, rownames.force = TRUE)
barplot(g_polarity)

## visualize sentiment
class(g_sentiment)
g_sentiment = data.matrix(g_sentiment, rownames.force = TRUE)
barplot(g_sentiment)

## break it down by sentence
fomc_sen = get_sentences(fomc_minutes)
fomc_sen

sentiment_vector = get_sentiment(fomc_sen, method = "syuzhet")
## A vector is a basic data structure in R.
## It is a sequence of elements that share the same data type. 
## (logical, integer, double, character, complex, or raw data)

sentiment_vector
summary(sentiment_vector)
boxplot(sentiment_vector)

## what was the most positive sentence in the paragraph?
max(sentiment_vector)
sentence_sentiment = data.frame(fomc_sen, sentiment_vector)
View(sentence_sentiment)
which.max(sentence_sentiment$sentiment_vector)

## we now have an index - what can we do with this?
most_positive = sentence_sentiment[which.max(sentence_sentiment$sentiment_vector),]
## don't forget the comma!

## Wordclouds
## look over to the right and see that these packages are checked
## if one is absent then go to the console below and type install.packages("MISSINGPACKAGE")
library("devtools")
library("RColorBrewer")
library("tm")
library("SnowballC")
library("wordcloud")

wordcloud(fomc_minutes,colors=c("blue","green"))
