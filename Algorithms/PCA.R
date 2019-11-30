setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library("FactoMineR")
library("corrplot")
library("factoextra")
library("fields")

source("mnist_reader.R")

colors <- gray.colors(255)

# load images
train_x = load_image_file("../Datasets/train-images-idx3-ubyte")
test_x  = load_image_file("../Datasets/t10k-images-idx3-ubyte")

# load labels
train_y = as.factor(load_label_file("../Datasets/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("../Datasets/t10k-labels-idx1-ubyte"))

test_x <- test_x / 255.0

#train_xy = cbind(train_x, train_y)
#test_xy = cbind(test_x, test_y)

# Compute PCA
resPCA <- PCA(test_x, scale.unit = FALSE)
label_col <- test_y

resPCA <- PCA(train_x, scale.unit = FALSE, ncp=50)
label_col <- train_y

resPCA <- PCA(rbind(train_x, test_x), scale.unit = FALSE)
label_col <- as.factor(c(test_y, train_y))

#layout(matrix(c(1,2), ncol=2))
plot.PCA(resPCA, choix = "var", label = "none")

fviz_pca_ind(resPCA, col.ind = label_col, label = "none", addEllipses = TRUE)
fviz_pca_var(resPCA, label = "none")

plot.PCA(resPCA, choix = "ind", col.ind = label_col, label = "none")

all_pred <- c()
index <- c()
data_y <- test_y
data_x <- test_x
for(y in unique(data_y)){
  conflict <- TRUE
  cpt <- 1
  while(conflict){
    ind <- which(y == data_y)[cpt]
    print(ind)
    pred <- predict.PCA(resPCA, data_x[ind,])$coord
    temp_pred <- pred[1:2]
    if(! is.null(all_pred)){
      dist <- apply(all_pred, MARGIN=1, function(x) { sqrt(sum((x - temp_pred)^2)) } )
      conflict <- sum(dist < 1050) != 0
      cpt <- cpt + 1
    }else{
      conflict <- FALSE 
    }
  }
  index <- c(index, ind)
  all_pred <- rbind(all_pred, temp_pred)
  print(temp_pred)
  add.image(pred[1], pred[2], matrix(rev(unlist(data_x[ind,])), nrow=28), image.width = 0.15, col=colors) 
}

# index <- c()
# for(y in unique(train_y)){
#   to_add <- which(train_y == y)[1]
#   index <- c(index, to_add)
# }
# for(ind in index){
#   print(ind)
#   pred <- predict.PCA(resPCA, train_x[ind,])$coord
#   print(pred[1:2])
#   add.image(pred[1], pred[2], matrix(rev(unlist(train_x[ind,])), nrow=28), image.width = 0.15, col=colors) 
# }


#var <- get_pca_var(resPCA)
#corrplot(var$cos2)
#corrplot(var$contrib, is.corr=FALSE)




# Images reconstruction
plot(resPCA$eig[,3])
which(resPCA$eig[,3] <= 80)

rec <- reconst(resPCA, ncp = 23)

#index <- c(2, 17, 6, 4, 20, 9, 19, 7, 24, 1)

# Select one obs for each class
rec <- rec[index,]

normalize <- function(x){
  (x - min(x)) / (max(x) - min(x)) * 255
}

#rec <- read.csv("decoded_images.csv", row.names = 1)

rec <- apply(rec, MARGIN = 1, FUN = normalize)

for(i in 1:length(index)){
  svg(filename=paste("../Results/pca_reconst_",i,".svg", sep=""), width = 20, height = 10)
  layout(matrix(c(1:2), ncol=2))
  temp <- matrix(rev(rec[,i]), nrow = 28)
  image(temp, col = colors)
  temp <- matrix(rev(as.double(data_x[index[i],])), nrow = 28)
  image(temp, col = colors) 
  dev.off()
  #readline(prompt="Press [enter] to continue")
}

temp <- matrix(rec[,1], nrow = 28, byrow = TRUE)
image.plot(temp, col = colors)
temp <- matrix(as.double(train_x[index[1],]), nrow = 28, byrow = TRUE)
image.plot(temp, col = colors) 
