setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library("Rtsne")
library("fields")
library("R.matlab")

source("mnist_reader.R")

plot_tsne <- function(tsne){
  colors = rainbow(length(unique(train_y)))
  names(colors) = unique(train_y)
  kl <- round(tsne$itercosts[length(tsne$itercosts)], digits = 2)
  plot_name <- paste("t-SNE with KL-divergence = ", kl, sep ="")
  plot_name <- NULL
  plot(tsne$Y, xlab = "Comp. 1", ylab = "Comp. 2", col = colors[train_y], main = plot_name)
}

# load images
train_x = load_image_file("../Datasets/train-images-idx3-ubyte")
test_x  = load_image_file("../Datasets/t10k-images-idx3-ubyte")

# load labels
train_y = as.factor(load_label_file("../Datasets/train-labels-idx1-ubyte"))
test_y = as.factor(load_label_file("../Datasets/t10k-labels-idx1-ubyte"))


train_x_enc = as.matrix(readMat("../Results/ae/ae_encoded.mat")$X)

ae = as.matrix(readMat("../Results/ae/ae_49_encoded.mat")$X)
ae2 = as.matrix(readMat("../Results/ae/ae_encoded.mat")$X)
ae3 = as.matrix(readMat("../Results/ae/ae_100_encoded.mat")$X)
cae = as.matrix(readMat("../Results/ae/cae1_encoded.mat")$X)
cae2 = as.matrix(readMat("../Results/ae/cae2_encoded.mat")$X)

all_data <- cbind(ae, ae2, ae3, cae, cae2)
dim(all_data)





#################################################
tsne <- Rtsne(train_x, verbose = TRUE) 
tsne <- Rtsne(test_x, verbose = TRUE) 
tsne <- Rtsne(train_x_enc, verbose = TRUE)
tsne <- Rtsne(all_data, verbose = TRUE) 

dev.off()
layout(matrix(1:2, nrow=1))
plot_tsne(tsne)
plot_tsne(tsne)

all_pred <- c()
index <- c()
data_y <- train_y
data_x <- train_x
for(y in unique(data_y)){
  conflict <- TRUE
  cpt <- 1
  while(conflict){
    ind <- which(y == data_y)[cpt]
    print(ind)
    pred <- tsne$Y[ind,]
    temp_pred <- pred[1:2]
    if(! is.null(all_pred)){
      dist <- apply(all_pred, MARGIN=1, function(x) { sqrt(sum((x - temp_pred)^2)) } )
      conflict <- sum(dist < 15) != 0
      cpt <- cpt + 1
    }else{
      conflict <- FALSE 
    }
  }
  index <- c(index, ind)
  all_pred <- rbind(all_pred, temp_pred)
  print(temp_pred)
  add.image(pred[1], pred[2], matrix(rev(unlist(data_x[ind,])), nrow=28), image.width = 0.15, col=gray.colors(255)) 
}
#################################################



















#################################################
# OLD
#################################################
train_xy = cbind(train_x, train_y)
test_xy = cbind(test_x, test_y)

label_col = 785

times <- c()

#perpl <- 5

perpl_v <- c(1:20 * 5, 1:10 * 25 + 100)

save_plot = FALSE

#for(i in c(1:10)){
for(perpl in perpl_v){
  # Run tSNE
  start_time <- Sys.time()
  tsne <- Rtsne(train_x, perplexity = perpl, eta = 200.0, verbose = TRUE) 
  stop_time <- Sys.time()
  time_taken <- stop_time - start_time
  times <- c(times, time_taken)
  
  # Plot
  if(save_plot){
    name <- paste("tsne_", perpl, ".png", sep = "")
    colors = rainbow(length(unique(train_y)))
    names(colors) = unique(train_y)
    kl <- round(tsne$itercosts[length(tsne$itercosts)], digits = 2)
    plot_name <- paste("t-SNE with KL-divergence = ", kl, sep ="")
    png(filename=name, width = 1000, height = 1000)
    plot(tsne$Y, xlab = "Comp. 1", ylab = "Comp. 2", col = colors[train_y], main = plot_name)
    dev.off() 
  }
  
  # Increase perplexity
  #perpl <- perpl +5
}

print(times)

png(filename="tSNE_time", width = 1000, height = 1000)
plot(perpl_v, times, type = "b", xlab = "Perplexity", ylab = "Computation time")
dev.off() 

kl_res <- c(4.09, 3.73, 3.48, 3.33, 3.18, 3.08, 2.99, 2.91, 2.83, 2.78, 2.73, 2.68, 2.62, 2.56, 2.54, 2.51, 2.47, 2.43, 2.39, 2.36, 2.21, 2.14, 2.03, 1.97, 1.88, 1.83, 1.77, 1.72, 1.67, 1.64)
perpl_v <- c(1:20 * 5, 1:10 * 25 + 100)
plot(perpl_v, kl_res, type = "b", xlab = "Perplexity", ylab = "KL divergence")

kl_res <- c(2.02, 1.85, 1.74, 1.65, 1.59, 1.57, 1.54, 1.53, 1.51)
learn_v <- c(50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)
plot(learn_v, kl_res, type = "b", xlab = "Learning rate", ylab = "KL divergence")
