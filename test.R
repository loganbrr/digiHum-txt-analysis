library(pdftools)
library(tm)
library(stringr)
library(quanteda)
#/ load libs and then create a var for path #
pdf_path <- 'assets/mar19-2025.pdf'

preprocess_file <- function(pdf_path){
  text <- pdf_text(pdf_path) #/ extract txt from PDF #
  text <- text[-length(text)] #/ last page is always irrelevant #
  full_text <- paste(text, collapse = ' ')
  full_text <- tolower(full_text)
  
  start_index <- str_locate(full_text, "recent indicators")[,1]
  if (!is.na(start_index)){ #/ txt before this sentence is irrelevant #
    full_text <- substr(full_text, start_index, nchar(full_text))
  } #/ the Feds voting are irrelevant to sentiment #
  full_text <- str_replace(full_text, "voting for.*?(?=\\b[a-z])", "")
  #/ as are media inquiries #
  full_text <- str_replace(full_text, "for media inquiries.*?(?=malto:)", "")
  
  #/ remove all uniquely irrelevant words to this data #
  remove_phrases <- c("committee", "agency", "treasury", "federal reserve", 
                      "federal open market committee", "board of governors", 
                      "voted", "open market desk", "federal reserve bank of new york", 
                      "system open market account", "policy", "securities", "percent")
  for (phrase in remove_phrases){
    full_text <- str_replace(full_text, phrase, '')
  }
  
  #/ remove dates, dollar values, basis points, whitespace #
  full_text <- str_replace_all(full_text, "\\b(january|february|march|april|may|june|july|august|september|october|november|december) \\d{1,2}, \\d{4}\\b", "")
  full_text <- str_replace_all(full_text,  "\\$[\\d,.]+", "")
  full_text <- str_replace_all(full_text, "\\d+-\\d+/\\d+", "")
  full_text <- str_replace_all(full_text, "[[:punct:]]", "")
  full_text <- str_squish(full_text)
  
  #/ remove stopwords with packages #
  corpus <- Corpus(VectorSource(full_text))
  stopwords <- setdiff(stopwords('en'), c('inflation', 'growth', 'rate', 'increase', 'decrease', 'stagnate'))
  corpus <- tm_map(corpus, removeWords, stopwords)
  cleaned_text <- sapply(corpus, as.character)
  return(cleaned_text)
}

cleaned_text <- preprocess_file(pdf_path)
print(cleaned_text) #/ apply function to my file and save as .txt #
writeLines(cleaned_text, "assets/cleaned_fomc.txt")