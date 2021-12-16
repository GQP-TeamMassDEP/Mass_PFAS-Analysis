library(datasets)
data(iris)

irisX <- iris[,1:4]
ncomp <- 2
pca_iris        <- prcomp(irisX , center=T, scale=T)
rawLoadings     <- pca_iris$rotation[,1:ncomp] %*% diag(pca_iris$sdev, ncomp, ncomp)
rotatedLoadings <- varimax(rawLoadings)$loadings
invLoadings     <- t(pracma::pinv(rotatedLoadings))
scores          <- scale(irisX) %*% invLoadings  # my scores from rotated loadings which are standardized

# want to use APCS to do MLR instead of these scores
#step 1: create artificial sample with zero concentrations for all variables  
z0i              <- matrix(-colMeans(irisX)/sqrt(apply(irisX, 2, var)), nrow = 1)
#step 2: find its rotated PC score by multiplying the transposed rotated loading from the original sample
scores0         <- as.numeric(z0i %*% invLoadings) # my absolute zero PC scores (supposedly...) 
#step 3: now to calculate my new "APCS"
scores0         <- matrix(rep((scores0), each = nrow(scores)),nrow = nrow(scores))
ACPS            <- scores - scores0
