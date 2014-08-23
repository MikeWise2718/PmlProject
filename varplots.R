
dfva <- read.csv("dftestAv-3it.csv")
dfva$varnum <- "All vars"

dfv3 <- read.csv("dftest3v-3it.csv")
dfv3$varnum <- "3 vars"

dfv6 <- read.csv("dftest6v-3it.csv")
dfv6$varnum <- "6 vars"

dfall <- merge(dfva,dfv3,all=T)
dfall <- merge(dfall,dfv6,all=T)

qplot(prob,acc,data=dfall) + geom_smooth(method=loess) + 
  facet_grid( . ~ varnum ) +
  ggtitle("Accuracy vs. Training Percentage") +
  labs(x="Training Percentage",y="Accuracy")

