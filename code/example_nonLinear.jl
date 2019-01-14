# Load X and y variable
using JLD, Random
data = load("../data/nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
global best_val = Inf
#println("Xtrain size", size(Xtrain))
for i = 0.00 : 0.001: 0.02
	for j = 0.85 : 0.01 : 1 
		lambda = i
		sigma = j
		valErrorSum = 0
		for k = 1:3
			# shuffle training set using random permutation
			shuffled = randperm(n)
			boundary = div(n, 2) # 100 in this case
			train = shuffled[1:boundary]
			val = shuffled[boundary+1:end]

			Xtrain = reshape(X[train], boundary, 1)
			ytrain = y[train]
			Xval = reshape(X[val], boundary, 1)
			yval = y[val]

			model = leastSquaresRBFL2(Xtrain,ytrain,lambda,sigma)
			yhat = model.predict(Xval)
			valErrorSum += sum((yhat - yval).^2)/size(Xval,1)
		end
		valError = valErrorSum/3
		if valError < best_val 
			global best_lambda = lambda
			global best_sigma = sigma
			global best_val = valError
		end
	end
end


# Report the error on the validation set
using Printf
@printf("Validation Error = %.2f\n",best_val)
@printf("Best lambda = %.3f\n",best_lambda)
@printf("Best sigma = %.3f\n",best_sigma)

model = leastSquaresRBFL2(Xtrain,ytrain,best_lambda,best_sigma)

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
