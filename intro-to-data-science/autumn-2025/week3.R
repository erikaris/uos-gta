# **************************************************************#
# R/Rstudio Practical Part 1
# **************************************************************#

# Load in a library called MASS to get access to more datasets
library(MASS)

# Load the "women" dataset that contains the height and weight of a sample of women
data()
data("women")
women

names(women)
str(women)
summary(women)

# Explore the distribution of a continuous variable using histogram
hist(women$height)

# View what you can modify in the histogram using the help page
?hist

# Modify the histogram
hist(x=women$height, breaks=4, main="Histogram showing women's heights", 
     xlab="Height") # I have put all the argument names in this example.

# Display the graph side by side, e.g., in this case, 1 row and 2 columns     
par(mfrow=c(1,2))
hist(x=women$height, breaks=4, main="Heights", xlab="Height")
hist(x=women$weight, breaks=2, main="Weights", xlab="Weight")

# To display one graph at a time you can use the command:
# par(mfrow=c(1,1))

plot(women)

# **************************************************************#
#  Importing data into R/RStudio
# **************************************************************#

# View the datasets available in R
data()

library(tidyverse)
testFile<-read_tsv("./test.tsv")
testFile

# By default, it reads the first line of the file as the header.
# To make it ignore the header, set the parameter col_names to FALSE.

testFile<-read_tsv("test.tsv", col_names=FALSE)
testFile

testCSVFile<-read_csv("freeschoolmeals.csv")
head(testCSVFile)

# You can impose the data type of each column using the col_types
# parameter

testCSVFile<-read_csv("freeschoolmeals.csv", col_types="cciici")
head(testCSVFile)

# You can use readxl to read Excel Files

library(readxl)
excelFile<-file.path("indicator hiv estimated prevalence% 15-49.xlsx")
testExcelFile<-read_excel(excelFile, sheet="Data")
head(testExcelFile)

# Exercise: Using the visualisations you saw in the previous section, explore
# the two datasets you have loaded: the free school meals and the HIV
# prevalence. For example, you can compare the distribution of HIV prevalence
# on different years using side by side histogram

########### Erika's solution 1 1: base R vis  ###############
# Pick two year columns
year1 <- "2000.0"
year2 <- "2010"

# Extract only Country + 2 selected year columns
hiv_two_years <- testExcelFile[, c("Estimated HIV Prevalence% - (Ages 15-49)", year1, year2)]

# Rename columns for convenience
names(hiv_two_years) <- c("Country", "Year1", "Year2")

# clean the data: remove NAs. 
# Keep only rows where at least one of the two years has a value
# hiv_two_years <- hiv_two_years[!is.na(hiv_two_years$Year1) & !is.na(hiv_two_years$Year2), ]
hiv_two_years <- hiv_two_years[!(is.na(hiv_two_years$Year1) & !is.na(hiv_two_years$Year2)), ]

# create side-by-side histograms
# Set up plotting area: 1 row, 2 columns
par(mfrow = c(1, 2))

# Histogram for Year 1
hist(hiv_two_years$Year1,
     main = paste("HIV Prevalence in", gsub("\\.0", "", year1)),
     xlab = "HIV Prevalence (%)",
     col = "skyblue", border = "white")

# Histogram for Year 2
hist(as.numeric(hiv_two_years$Year2),
     main = paste("HIV Prevalence in", year2),
     xlab = "HIV Prevalence (%)",
     col = "salmon", border = "white")


#########  Erika's solution 2: ggplot version  ############
library(tidyverse)

# Pick multiple year columns
years <- c("2000.0", "2001.0", "2002.0", "2003.0", "2004.0")

# Extract Country + selected years
hiv_multi_years <- testExcelFile[, c("Estimated HIV Prevalence% - (Ages 15-49)", years)]

# Rename columns for convenience
names(hiv_multi_years) <- c("Country", "Y2000", "Y2001", "Y2002", "Y2003", "Y2004")

# Convert all year columns to numeric
hiv_multi_years[,-1] <- lapply(hiv_multi_years[,-1], as.numeric)

# Remove rows where all selected years are NA
hiv_multi_years <- hiv_multi_years[rowSums(is.na(hiv_multi_years[,-1])) < length(years), ]

# Reshape to long format for ggplot
hiv_long <- pivot_longer(hiv_multi_years, 
                         cols = -Country,
                         names_to = "Year",
                         values_to = "Prevalence")

# histogram
ggplot(hiv_long, aes(x = Prevalence, fill = Year)) +
  geom_histogram(bins = 15, color = "white") +
  facet_wrap(~ Year, ncol = 2, scales = "free_y") +
  # facet_wrap(~ Year, ncol = 2) +
  labs(title = "Distribution of HIV Prevalence by Year",
       x = "HIV Prevalence (%)",
       y = "Number of Countries") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

# boxplot
ggplot(hiv_long, aes(x = Prevalence, y = Year, fill = Year)) +
  geom_boxplot() +
  # geom_jitter(width = 0.2, alpha = 0.5, color = "black") +  # show individual points
  labs(title = "HIV Prevalence by Year",
       x = "HIV Prevalence (%)",
       y = "Year") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")




# **************************************************************#
#  Read HTML and XML data
# **************************************************************#

install.packages("rvest")
library(rvest)

url <- "https://en.wikipedia.org/wiki/Sheffield"
wikiPage <- read_html(url)
wikiPage

h2Sections <- wikiPage %>% html_nodes("h2")
h2Sections

h2Sections[1]
h2Sections[2]
h2Sections[1:2]

h2Sections %>% html_text()

pageText <- wikiPage %>%
  html_nodes("p") %>%
  html_text()

pageText[1]

pageText[2]

# **************************************************************#
#  Erika's addition
# **************************************************************#

# 1. Read the Wikipedia page
url <- "https://en.wikipedia.org/wiki/Sheffield"
wikiPage <- read_html(url)

# 2. Extract all h2 headers (main sections)
h2Sections <- wikiPage %>% html_elements("h2") %>% html_text()
h2Sections

# 3. Extract all paragraphs (<p>)
paragraphs <- wikiPage %>% html_elements("p") %>% html_text()
head(paragraphs, 5)   # show first 5 paragraphs

# 4. Extract all links (<a>) and their URLs
links <- wikiPage %>% html_elements("a")
link_text <- links %>% html_text()
link_href <- links %>% html_attr("href")

# 5. Combine into a tibble
link_data <- tibble(text = link_text, url = link_href)
head(link_data, 10)  # show first 10 links

# 6. Extract all tables (<table>) on the page
tables <- wikiPage %>% html_elements("table")
length(tables)        # how many tables exist

# 7. Convert the first table to a data frame
if(length(tables) > 0){
  table1 <- tables[[1]] %>% html_table(fill = TRUE)
  head(table1)
}

# 8. Extract all images (<img>) and their sources
images <- wikiPage %>% html_elements("img")
img_src <- images %>% html_attr("src")
head(img_src, 10)  # show first 10 image URLs

# 9. Extract all h3 headers (subsections)
h3Sections <- wikiPage %>% html_elements("h3") %>% html_text()
h3Sections

# 10. Optional: Clean up text by trimming whitespace
library(stringr)
paragraphs_clean <- str_squish(paragraphs)
h2Sections_clean <- str_squish(h2Sections)

# 11. Example: Extract paragraphs under the first h2 section
first_h2 <- wikiPage %>% html_elements("h2") %>% .[1]
first_h2_paras <- first_h2 %>% html_elements(xpath = "following-sibling::p") %>% html_text()
head(first_h2_paras, 3)



# **************************************************************#
#  Read JSON data
# **************************************************************#
install.packages("jsonlite")
library(jsonlite)

json <- '[
          {"Name": "Mario", "Age": 32, "Occupation": "Plumber"},
          {"Name": "Peach", "Age": 21, "Occupation": "Princess"},
          {},
          {"Name": "Bowser", "Occupation": "Koopa"}
        ]'

mydf <- fromJSON(json)
mydf

myjson <- toJSON(mydf)
myjson

# read it in a pretty way
toJSON(fromJSON(json), pretty = TRUE, auto_unbox = TRUE)

citibike <- fromJSON("https://gbfs.citibikenyc.com/gbfs/en/station_information.json")
View(citibike)   # this will return error
str(citibike$data) # inspect citibike structure. 

View(citibike$data$stations)

stations <- citibike$data$stations$name
stations

### Using the visualisations you have learned show the distribution of available
### bikes and compare the number of available docks to the number of total docks
### (notice these are columns inside a dataframe that is inside a list).

# Select and clean numeric columns from citibike data
citibike_flat <- fromJSON("https://gbfs.citibikenyc.com/gbfs/en/station_information.json", flatten = TRUE) #flatten the data
str(citibike_flat)
stations_flat <- citibike_flat$data$stations
stations_flat

# Just keep relevant columns
stations_plot <- stations_flat %>%
  select(name, capacity, has_kiosk)

# Distribution of total docks (capacity)
ggplot(stations_plot, aes(x = capacity)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Station Capacities",
       x = "Number of Docks (Capacity)",
       y = "Count of Stations") +
  theme_minimal()

# Compare stations with vs without kiosk by capacity
ggplot(stations_plot, aes(x = capacity, fill = has_kiosk)) +
  geom_histogram(position = "dodge", binwidth = 5) +
  labs(title = "Station Capacities by Kiosk Availability",
       x = "Number of Docks (Capacity)",
       y = "Count of Stations",
       fill = "Has Kiosk") +
  theme_minimal()


## additional info
getOption("max.print")  # check default elements that R will print in console
options(max.print = 1000) # set maximum number of elements to print 1,000
