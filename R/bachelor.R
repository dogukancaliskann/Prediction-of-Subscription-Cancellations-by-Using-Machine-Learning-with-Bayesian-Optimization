library(readxl)
library(dplyr)
library(lubridate)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(caret)
library(randomForest)
library(e1071)
library(pROC)
library(scales)
library(sjmisc)
library(sjlabelled)
library(plyr)
library(remotes)
library(naniar)
library(mice)
library(VIM)
library(cowplot)
library(dplyr)
library(tidyr)


data <- read.csv("C:/Membership woes.csv", header = TRUE,na.strings=c("","NA"),
                 colClasses = c("character", "numeric", "numeric", "character", "character", "numeric", "character", 
                                "character", "numeric", "numeric", "character", "character", "character", "character", "character"))

head(data)

# Train-Test split #
index <- sample(2, nrow(data),replace = T,prob = c(0.6,0.4))
data_train <- data.frame(data[index==1,])
data_test <- data.frame(data[index==2,])
######


dim(data) #Training Data dimensions

##data train -> df_data
df_data <- data.frame(data)
remove(data_train)

summary(df_data)

str(df_data)

# Identifying Categorical and Numeric Variables #
df_data = df_data %>% mutate_if(is.character, as.factor)
cat_var <- names(df_data)[which(sapply(df_data, is.factor))]
print('Categorical Variables:')
print(cat_var)

numeric_var <- names(df_data)[which(sapply(df_data, is.numeric))]
print('Numeric Variables')
print(numeric_var)

# Variable Transformation and Creation #
df_data$START_DATE..YYYYMMDD. <- as.Date(df_data[["START_DATE..YYYYMMDD."]],"%Y%m%d")
df_data$END_DATE...YYYYMMDD. <- as.Date(df_data[["END_DATE...YYYYMMDD."]],"%Y%m%d")
df_data$START_YEAR <- (year(df_data$START_DATE..YYYYMMDD.))
df_data$START_MONTH <- as.factor(month(df_data$START_DATE..YYYYMMDD.))

df_data$AGE_GROUP[df_data$MEMBER_AGE_AT_ISSUE >=0 & df_data$MEMBER_AGE_AT_ISSUE < 20] <- "<20 years"
df_data$AGE_GROUP[df_data$MEMBER_AGE_AT_ISSUE >=20 & df_data$MEMBER_AGE_AT_ISSUE < 40] <- "20-40 years"
df_data$AGE_GROUP[df_data$MEMBER_AGE_AT_ISSUE >=40 & df_data$MEMBER_AGE_AT_ISSUE < 60] <- "40-60 years"
df_data$AGE_GROUP[df_data$MEMBER_AGE_AT_ISSUE >=60 & df_data$MEMBER_AGE_AT_ISSUE < 80] <- "60-80 years"
df_data$AGE_GROUP[df_data$MEMBER_AGE_AT_ISSUE >=80 ] <- ">80 years"

df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=0 & df_data$MEMBERSHIP_TERM_YEARS < 20] <- "<20 years"
df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=20 & df_data$MEMBERSHIP_TERM_YEARS < 40] <- "20-40 years"
df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=40 & df_data$MEMBERSHIP_TERM_YEARS < 60] <- "40-60 years"
df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=60 & df_data$MEMBERSHIP_TERM_YEARS < 80] <- "60-80 years"
df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=80 & df_data$MEMBERSHIP_TERM_YEARS < 100] <- "80-100 years"
df_data$TERM_GROUP[df_data$MEMBERSHIP_TERM_YEARS >=100] <- ">100 years"

#Modeling for target as the churn indicator
df_data$TARGET[df_data$MEMBERSHIP_STATUS == "INFORCE"] <-"No"
df_data$TARGET[df_data$MEMBERSHIP_STATUS == "CANCELLED"] <- "Yes"

head(df_data)

# Univariate analysis of Categorical Variables #
table(df_data$MEMBERSHIP_STATUS)
table(df_data$MEMBER_MARITAL_STATUS)
table(df_data$MEMBER_GENDER)
table(df_data$MEMBER_OCCUPATION_CD)
table(df_data$PAYMENT_MODE)
table(df_data$MEMBERSHIP_PACKAGE)
table(df_data$MEMBER_MARITAL_STATUS)
table(df_data$AGE_GROUP)
table(df_data$TERM_GROUP)

# Univariate analysis of Numeric Variables using Histograms #
ggplot(gather(df_data[,c(numeric_var)]),aes(value))+geom_histogram(bins = 30)+facet_wrap(~key, scales = 'free_x')
par(mfrow=c(1,3))
hist(df_data$START_DATE..YYYYMMDD.,"years", xlab ='START_DATE') #No outliers
hist(df_data$END_DATE...YYYYMMDD.,"years", xlab ='END_DATE') #No outliers

par(mfrow=c(3,3))
boxplot(df_data$MEMBERSHIP_TERM_YEARS, main='MEMBERSHIP_TERM_YEARS') #No outliers
boxplot(df_data$MEMBER_AGE_AT_ISSUE, main='MEMBER_AGE_AT_ISSUE') #No outliers
boxplot(df_data$MEMBER_ANNUAL_INCOME, main='MEMBER_ANNUAL_INCOME') #Outliers present
boxplot(df_data$ADDITIONAL_MEMBERS, main='ADDITIONAL_MEMBERS') #No outliers
boxplot(df_data$ANNUAL_FEES, main='ANNUAL_FEES') #Outliers present
boxplot(df_data$START_DATE..YYYYMMDD., main='START_DATE')
boxplot(df_data$END_DATE...YYYYMMDD., main='END_DATE')

# Outlier Treatment #

remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- caps[1]
  y[x > (qnt[2] + H)] <- caps[2]
  y
}

# Capping Outliers outside 1.5 times the InterQuaritle range for Annual Income and Annual Fees at 5th and 95th percentile values
df_data$MEMBER_ANNUAL_INCOME <- remove_outliers(df_data$MEMBER_ANNUAL_INCOME)
df_data$ANNUAL_FEES <- remove_outliers(df_data$ANNUAL_FEES)

par(mfrow=c(1,2))
boxplot(df_data$MEMBER_ANNUAL_INCOME)
boxplot(df_data$ANNUAL_FEES)

############################# Missing Values Analysis ######################

sapply(df_data, function(x) round((sum(is.na(x))/length(x)*100),2))
colSums(sapply(df_data, is.na))

# Marital Status, Gender, Occupation Code and Income have missing values
# End Date has missing values which correspond to the count of cancelled memberships, hence valid

plot_Missing <- function(data_in, title = NULL){
  temp_df <- as.data.frame(ifelse(is.na(data_in), 0, 1))
  temp_df <- temp_df[,order(colSums(temp_df))]
  data_temp <- expand.grid(list(x = 1:nrow(temp_df), y = colnames(temp_df)))
  data_temp$m <- as.vector(as.matrix(temp_df))
  data_temp <- data.frame(x = unlist(data_temp$x), y = unlist(data_temp$y), m = unlist(data_temp$m))
  ggplot(data_temp) + geom_tile(aes(x=x, y=y, fill=factor(m))) + scale_fill_manual(values=c("white", "black"), name="Missing\n(0=Yes, 1=No)") + theme_light() + ylab("") + xlab("") + ggtitle(title)
}

plot_Missing(df_data[,colSums(is.na(df_data)) > 0])

df_data=df_data %>% mutate_if(is.factor, as.character)



df_data$MEMBER_MARITAL_STATUS[is.na(df_data$MEMBER_MARITAL_STATUS)] <- "Others"
df_data$MEMBER_GENDER[is.na(df_data$MEMBER_GENDER)] <- "Others"
df_data$MEMBER_OCCUPATION_CD[is.na(df_data$MEMBER_OCCUPATION_CD)] <- "Others"
df_data=df_data %>% mutate_if(is.character, as.factor)



colSums(sapply(df_data, is.na))
str(df_data)



churnozet=table(df_data$TARGET)
churnozet=data.frame(churnozet)
colnames(churnozet)=c("Churn","Freq")
perc=churnozet$Freq/sum(churnozet$Freq) 
churnozet$perc=perc
churnozet$percentage=percent(perc)

ggplot(churnozet, aes(x = "", y = perc, fill = Churn)) +
  geom_col(color = "black") +
  geom_label(aes(label = percentage), color = c("black", "black"),
             position = position_stack(vjust = 0.5),
             show.legend = FALSE) +
  guides(fill = guide_legend(title = "Churn")) +
  scale_fill_viridis_d() +
  coord_polar(theta = "y") + 
  scale_fill_manual(values = c("orange", "pink"))+
  theme_void()




# Bivariate analysis of some Categorical Variables #
par(mfrow=c(3,3))
plot(df_data$MEMBER_GENDER,df_data$MEMBERSHIP_STATUS)
plot(df_data$MEMBER_MARITAL_STATUS,df_data$MEMBERSHIP_STATUS)
plot(df_data$MEMBER_OCCUPATION_CD,df_data$MEMBERSHIP_STATUS)
plot(df_data$MEMBERSHIP_PACKAGE,df_data$MEMBERSHIP_STATUS)
plot(df_data$PAYMENT_MODE,df_data$MEMBERSHIP_STATUS)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$START_MONTH)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$START_YEAR)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$AGE_GROUP)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$TERM_GROUP)

# Bivariate analysis of some Numeric Variables #
par(mfrow=c(2,3))
plot(df_data$MEMBERSHIP_STATUS ~ df_data$ANNUAL_FEES)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$MEMBERSHIP_TERM_YEARS)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$MEMBER_ANNUAL_INCOME)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$MEMBER_AGE_AT_ISSUE)
plot(df_data$MEMBERSHIP_STATUS ~ df_data$ADDITIONAL_MEMBERS)


# Variable Selection #
correlationMatrix <- cor(df_data[,sapply(df_data,is.numeric)])
corrplot(correlationMatrix, method = "number", type = "upper", tl.cex= 0.8)
# Variables are very loosely correlated, hence no need to drop any
# Removing END_DATE as it directly correlates to the event, MEMBERSHIP_NUMBER as it is a unique identifier, AGENT_CODE as it has too many unique values
variablesToDrop <- names(df_data) %in% c('MEMBERSHIP_TERM_YEARS','START_DATE..YYYYMMDD.','MEMBERSHIP_NUMBER','AGENT_CODE','MEMBER_AGE_AT_ISSUE','MEMBERSHIP_STATUS','END_DATE...YYYYMMDD.')
newdata <- df_data[!variablesToDrop]
head(newdata)

newdata2 <- newdata


#Kategorilerin labellarý 
get_labels(newdata2$MEMBER_GENDER) #4 tane
get_labels(newdata2$MEMBER_MARITAL_STATUS) #Niþanlýya falan others dendi 5 tane var
get_labels(newdata2$MEMBER_OCCUPATION_CD) 
get_labels(newdata2$MEMBERSHIP_PACKAGE)
get_labels(newdata2$PAYMENT_MODE) #5
get_labels(newdata2$START_MONTH) #12 ay faktör (aralýk getir)
get_labels(newdata2$AGE_GROUP) #5
get_labels(newdata2$TERM_GROUP) #6
#target




#Label Encoding

replace=function(x){
  tur= x["MEMBER_GENDER"]
  if(tur=="F"){
    return(1)
  }else if(tur=="M"){
    return(2)
  }else if(tur=="Others"){
    return(3)
  }
}

a=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$MEMBER_GENDER=a

replace=function(x){
  tur= x["MEMBER_MARITAL_STATUS"]
  if(tur=="D"){
    return(1)
  }else if(tur=="M"){
    return(2)
  }else if(tur=="S"){
    return(3)
  }else if(tur=="W"){
    return(4)
  }else if(tur=="Others"){
    return(5)
  }
}

b=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$MEMBER_MARITAL_STATUS=b



newdata2$MEMBER_OCCUPATION_CD <- as.character(newdata2$MEMBER_OCCUPATION_CD)
newdata2$MEMBER_OCCUPATION_CD[which(newdata2$MEMBER_OCCUPATION_CD=="Others")] <- "3"
newdata2$MEMBER_OCCUPATION_CD[which(newdata2$MEMBER_OCCUPATION_CD=="4")] <- "3"
newdata2$MEMBER_OCCUPATION_CD[which(newdata2$MEMBER_OCCUPATION_CD=="5")] <- "3"
newdata2$MEMBER_OCCUPATION_CD[which(newdata2$MEMBER_OCCUPATION_CD=="6")] <- "3"
newdata2$MEMBER_OCCUPATION_CD <- as.numeric(newdata2$MEMBER_OCCUPATION_CD)

newdata2$MEMBER_OCCUPATION_CD[is.na(newdata2$MEMBER_OCCUPATION_CD)] <- "3"

newdata2$MEMBER_OCCUPATION_CD = as.numeric(newdata2$MEMBER_OCCUPATION_CD)




replace=function(x){
  tur= x["MEMBERSHIP_PACKAGE"]
  if(tur=="TYPE-A"){
    return(1)
  }else if(tur=="TYPE-B"){
    return(2)
  }
}

c=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$MEMBERSHIP_PACKAGE=c



replace=function(x){
  tur= x["PAYMENT_MODE"]
  if(tur=="ANNUAL"){
    return(1)
  }else if(tur=="MONTHLY"){
    return(2)
  }else if(tur=="QUARTERLY"){
    return(3)
  }else if(tur=="SEMI-ANNUAL"){
    return(4)
  }else if(tur=="SINGLE-PREMIUM"){
    return(5)
  }
}


d=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$PAYMENT_MODE=d



newdata2$START_MONTH <- as.character(newdata2$START_MONTH)
newdata2$START_MONTH[which(newdata2$START_MONTH=="12")] <- "1"
newdata2$START_MONTH[which(newdata2$START_MONTH=="2")] <- "1"

newdata2$START_MONTH[which(newdata2$START_MONTH=="3")] <- "2"
newdata2$START_MONTH[which(newdata2$START_MONTH=="4")] <- "2"
newdata2$START_MONTH[which(newdata2$START_MONTH=="5")] <- "2"

newdata2$START_MONTH[which(newdata2$START_MONTH=="6")] <- "3"
newdata2$START_MONTH[which(newdata2$START_MONTH=="7")] <- "3"
newdata2$START_MONTH[which(newdata2$START_MONTH=="8")] <- "3"


newdata2$START_MONTH[which(newdata2$START_MONTH=="9")] <- "4"
newdata2$START_MONTH[which(newdata2$START_MONTH=="10")] <- "4"
newdata2$START_MONTH[which(newdata2$START_MONTH=="11")] <- "4"


newdata2$START_MONTH <- as.numeric(newdata2$START_MONTH)



replace=function(x){
  tur= x["AGE_GROUP"]
  if(tur=="<20 years"){
    return(1)
  }else if(tur=="20-40 years"){
    return(2)
  }else if(tur=="40-60 years"){
    return(3)
  }else if(tur=="60-80 years"){
    return(4)
  }else if(tur=="80-100 years"){
    return(5)
  }else if(tur==">100 years"){
    return(6)
  }
}

e=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$AGE_GROUP=d



replace=function(x){
  tur= x["TERM_GROUP"]
  if(tur=="<20 years"){
    return(1)
  }else if(tur=="20-40 years"){
    return(2)
  }else if(tur=="40-60 years"){
    return(3)
  }else if(tur=="60-80 years"){
    return(4)
  }else if(tur=="80-100 years"){
    return(5)
  }else if(tur==">100 years"){
    return(6)
  }
}

f=apply(newdata2, MARGIN=1, FUN = replace)
newdata2$TERM_GROUP=f


newdata2 <- newdata2 %>%
  mutate(TARGET = ifelse(TARGET == "No",0,1)) #Target da dönüþtü





mcar_test(newdata2[,c(1,4)]) #Ýkili dene

#Pattern leri karþýlaþtýr düzenlenmiþ ve düzenlenmemiþ verilerde
aggr_plot_rawdata <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                          labels=names(data), cex.axis=.7, gap=3, 
                          ylab=c("Histogram of missing data in Raw Data","Pattern"))



aggr_plot_ <- aggr(newdata2, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                   labels=names(newdata2), cex.axis=.7, gap=3, 
                   ylab=c("Histogram of missing data in Processed Data","Pattern"))




#Mice Ýmputation


mice_imputed <- data.frame(
  original = newdata2$MEMBER_ANNUAL_INCOME,
  imputed_pmm = complete(mice(newdata2, method = "pmm"))$MEMBER_ANNUAL_INCOME,
  imputed_cart = complete(mice(newdata2, method = "cart"))$MEMBER_ANNUAL_INCOME,
  imputed_lasso = complete(mice(newdata2, method = "lasso.norm"))$MEMBER_ANNUAL_INCOME
)

mice_imputed


h1 <- ggplot(mice_imputed, aes(x = original)) +
  geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") +
  ggtitle("Original distribution") +
  theme_classic()
h2 <- ggplot(mice_imputed, aes(x = imputed_pmm)) +
  geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") +
  ggtitle("PMM-imputed distribution") +
  theme_classic()
h3 <- ggplot(mice_imputed, aes(x = imputed_cart)) +
  geom_histogram(fill = "#1543ad", color = "#000000", position = "identity") +
  ggtitle("CART-imputed distribution") +
  theme_classic()
h4 <- ggplot(mice_imputed, aes(x = imputed_lasso)) +
  geom_histogram(fill = "#ad8415", color = "#000000", position = "identity") +
  ggtitle("Lasso-imputed distribution") +
  theme_classic()

plot_grid(h1, h2, h3, h4, nrow = 2, ncol = 3)



#MissForest Ýmputation
library(missForest)


missForest_imputed <- data.frame(
  original = newdata2$MEMBER_ANNUAL_INCOME,
  imputed_missForest = missForest(newdata2)$ximp$MEMBER_ANNUAL_INCOME
)

missForest_imputed

h5 <- ggplot(missForest_imputed, aes(x = original)) +
  geom_histogram(fill = "#ad1538", color = "#000000", position = "identity") +
  ggtitle("Original distribution") +
  theme_classic()

h6 <- ggplot(missForest_imputed, aes(x = imputed_missForest)) +
  geom_histogram(fill = "#15ad4f", color = "#000000", position = "identity") +
  ggtitle("missForest-imputed distribution") +
  theme_classic()

plot_grid(h5, h6, nrow = 1)

#PMM yi seçiyoruz.


aggr_plot_rawdata <- aggr(data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data in Raw Data","Pattern"))



aggr_plot_ <- aggr(newdata2, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                          labels=names(newdata2), cex.axis=.7, gap=3, 
                          ylab=c("Histogram of missing data in Processed Data","Pattern"))




write.csv(mice_imputed, "C:/Users/dogukan1/Desktop/miceimputed.csv", row.names=FALSE)














