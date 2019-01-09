include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(w,X,y)
	(n,d) = size(X)
	k = maximum(y)
	W = reshape(w, k, d)

	# objective function
	f = 0
	for i = 1:n
		f += log.(sum(exp.(W*X[i, :])))-dot(W[y[i], :], X[i, :])
	end

	# gradient
	G = zeros(k,d)
	prob = zeros(k)
	for c = 1:k
		for j = 1:d
			for i = 1:n
				num = exp(dot(W[c, :], X[i, :]))
				denom = sum(exp.(W*X[i, :]))
				prob[c] = num/denom
				if c == y[i]
					prob[c] -= 1
				end
				G[c, j] += X[i, j] * prob[c]
			end
		end
	end
	g = reshape(G, k*d)
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function softmax(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(k,d)
	w = reshape(W, k*d)

	funObj(w) = softmaxObj(w,X,y)

	w = findMin(funObj,w,derivativeCheck=true,verbose=false)
	W = reshape(w, k, d)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W',dims=2)

	return LinearModel(predict,W)
end