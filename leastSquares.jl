include("misc.jl")
using LinearAlgebra

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function rbfBasis(X1, X2, sigma)
	# pass Xtest, X will generate Ztest
	# pass X, X will generate Z
	# form Z
	Z = exp.(-distancesSquared(X1,X2)./(2sigma^2))

	return Z
end

function leastSquaresRBFL2(X,y,lambda, sigma)

	n = size(X,1)
	Z = rbfBasis(X, X, sigma)
	# Find regression weights minimizing squared error
	w = (Z'*Z + lambda*Matrix{Float64}(I, n, n))\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = rbfBasis(Xtilde, X, sigma)*w

	# Return model
	return LinearModel(predict,w)
end