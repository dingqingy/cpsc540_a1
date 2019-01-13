# Load X and y variable
using JLD
data = load("../data/nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# shuffle training set using random permutation
using Random
shuffled = randperm(n)
boundary = div(n, 2) # 100 in this case
train = shuffled[1:boundary]
val = shuffled[boundary+1:end]

Xtrain = reshape(X[train], boundary, 1)
ytrain = y[train]
Xval = reshape(X[val], boundary, 1)
yval = y[val]

# Fit least squares model
include("leastSquares.jl")
lambda = 1
sigma = 1
println("Xtrain size", size(Xtrain))
model = leastSquaresRBFL2(Xtrain,ytrain,lambda,sigma)


# Report the error on the validation set
using Printf
t = size(Xval,1)
yhat = model.predict(Xval)
valError = sum((yhat - yval).^2)/t
@printf("Validation Error = %.2f\n",valError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
