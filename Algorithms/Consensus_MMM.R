setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(R.matlab)
library(mclust)
library(Rmixmod)
library(aricode)
library(klaR)

source("mnist_reader.R")


#################################################
# Import data
#################################################
res_path = "../Results/ae/"
ae = readMat(paste(res_path, "ae_49_encoded.mat", sep=""))
ae2 = readMat(paste(res_path, "ae_encoded.mat", sep=""))
ae3 = readMat(paste(res_path, "ae_100_encoded.mat", sep=""))
cae = readMat(paste(res_path, "cae1_encoded.mat", sep=""))
cae2 = readMat(paste(res_path, "cae2_encoded.mat", sep=""))

res_models <- read.csv("../Results/ae/partitions_5.csv")
dim(res_models)

label <- as.factor(load_label_file("../Datasets/train-labels-idx1-ubyte"))

#################################################
# II
#################################################

nmis <- c()
aris <- c()
for(i in 1:20){
  
  #res_mixmod <- mixmodCluster(as.data.frame(type.convert(res_models)), nbCluster = 10, models=mixmodMultinomialModel(), dataType="qualitative")
  #pred <- res_mixmod@bestResult@partition
  
  res_kmodes <- klaR::kmodes(res_models, 10, iter.max = 3000) #, fast = F)
  pred <- res_kmodes$cluster
  
  res_nmi <- NMI(as.factor(pred), as.factor(label))
  res_ari <- ARI(as.factor(pred), as.factor(label))
  
  cat(res_nmi, ", ", res_ari, "\n")
  
  nmis <- c(nmis, res_nmi)
  aris <- c(aris, res_ari)
}

cat("##############\n# Consensus\n##############\n")
cat(mean(nmis), " $\\pm$ ", sd(nmis), "\n")
cat(mean(aris), " $\\pm$ ", sd(aris), "\n")
