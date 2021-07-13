library(orca)
library("FNN")

setwd("C:/Users/albil/Documents/GitHub/OCLRecommendation")

for (i in 1:318) {
  nodes_loc <- sprintf("C:/Users/albil/Documents/GitHub/OCLRecommendation/GraphletsNodes/nodelist%s.txt", i)
  edges_loc <- sprintf("C:/Users/albil/Documents/GitHub/OCLRecommendation/GraphletsEdges/edgelist%s.txt", i)
  csv_loc <- sprintf("C:/Users/albil/Documents/GitHub/OCLRecommendation/GraphletsCSV/graphlets_model%s.csv", i)
  nodes <- scan(nodes_loc,what="", sep="\n")
  edges <- try(read.table(edges_loc))
  if (inherits(edges, 'try-error')) next
  orbits <- count5(edges)
  orbits_df <- tail(orbits, n=(strtoi(max(nodes))-strtoi(min(nodes)))+1)
  write.csv(orbits_df, csv_loc,row.names = TRUE)
}
