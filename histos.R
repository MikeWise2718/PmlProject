
d <- melt(diamonds[,-c(2:4)])
ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram(binwidth=range/30)

