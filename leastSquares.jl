include("misc.jl")
using GLPKMathProgInterface, LinearAlgebra, MathProgBase

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)
	println("size of w", size(w))

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

function leastAbsolute(X,y)

	# Add bias column
	n, d = size(X)
	Z = [ones(n,1) X]

	# use linear programming to find solution
	c = [zeros(d+1);ones(n)]
	eye = Matrix{Float64}(I, n, n)
	A = [Z eye; Z -eye]
	b = [ones(n)*Inf; y]
	b = reshape(b, 2n)
	d = [y; ones(n)*-Inf]
	d = reshape(d, 2n)
	solution = linprog(c,A,d,b,-Inf,Inf,GLPKSolverLP())
	x = solution.sol
	w = [x[1]; x[2]]

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end