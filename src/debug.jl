
using Revise
using InplaceOps
#using LinearMaps
using LinearMapsAA
using Test

# LinearMapAA_test_kron()


N = 4
Mat = ones(6,N)

A1 = LinearMapAA(x -> Mat*x, (6, N))
A2 = LinearMapAA(x -> Mat*x, (6, N) ; odim=(3,2))

A1 = LinearMapAA(Mat)
A2 = LinearMapAA(Mat ; odim=(3,2))

#A0 = LinearMap(Mat)

u = ones(6) 
#u' * A0

#X = ones(N,2)
#Y = A2 * X
X = ones(A2._odim..., 2)
Y = X * A2

LinearMapAA_test_vmul(A1)
LinearMapAA_test_vmul(A2)

#=
#A1 = A2

x = ones(N) 
X = ones(N,2,3)

for A in (A1,A2)
	A * x
	y = A * x
	mul!(y, A, x)
	mul!(y, A, x, 1, 0)

	if A isa LinearMapAO
		Y = A * X
		mul!(Y, A, X)
	#	LinearMapsAA.mul!(Y, A, X)
		mul!(Y, A, X, 1, 0)
	#	LinearMapsAA.mul!(Y, A, X, 1, 0)
		@! Y = A * X
		@which *(A, X)
	end

	@which *(A, x)
end

=#
