using MKL, LinearAlgebra, Plots, Base.Threads, HDF5


# HDF5 functions # Delete the dataset if it exists
function del_if_exists(f, dataset::String)
        if haskey(f, dataset) 
                delete_object(f, dataset)
        end
end

# HDF5 save data to dataset
function save_data(f, dataset::String, data)
        del_if_exists(f, dataset)
        f[dataset] = data
end


# initialize the coefficient matrix
function init_coeff_matrix()
	# k denotes a basis state, r an index in the diagonal representation
	# k=1 : down spin 
	# k=2 : empty state	
	# k=3 : up spin
	# k=4 : doubly occupied state

	# M : constant matrix of coefficients:
	# M(1,k,kp) = <k| c_Up |kp>
	# M(2,k,kp) = <k| c_Down |kp>

	M = zeros(Float64,2,4,4)
	M[1,2,3] = 1
	M[1,1,4] = 1
	M[2,2,1] = 1
	M[2,3,4] = -1

	return M
end


# Wilson chain of a flat band hybridization function
function init_wilson_chain(Lambda::Float64, lmax::Int, V::Float64)
	# correction factor
	aLambda = 0.5 * (Lambda +1)/(Lambda -1) * log(Lambda)

	# Wilson chain parameters
	tW = zeros(Float64, lmax+2) # +2 is for site 0 and imp (-1)
	tW[1] = sqrt(aLambda/Lambda) * V # tW[1] technically not part of the Wilson chain

	@inbounds for l in 2:length(tW)
	tW[l] = 0.5 * (1+ (1/Lambda)) * (1- Lambda^(-l +1)) / 
		sqrt( (1- Lambda^(-2l +1)) * (1- Lambda^(-2l +3)) )
	end

	return tW
end

# Hamiltonian of first iteration
function initialize_ham(eps::Float64, U::Float64, Lambda::Float64, 
		M::Array{Float64})
	eps /= Lambda
	U /= Lambda

	# we start with imp Hilbert space of length 4
	rmax::Int = 4 # number of states in the diagonal representation of H

	energies = zeros(Float64, rmax)
	energies[1] = eps # down spin
	energies[2] = 0. # empty
	energies[3] = eps # up spin
	energies[4] = 2*eps + U # doubly occupied


	# Matrix Elements:  P(s,r,rp) = <r| c_s^dagger |rp>, s=spin
	P = zeros(Float64, 2, rmax, rmax)

	# here we only have one site, 
	# so the basis states are just the diagonal states
	@threads for s in 1:2
		P[s,:,:] = transpose(view(M,s,:,:)) # spin up, down
	end

	return rmax, energies, P
end


# update Hamiltonian for added site
function new_ham(rmax::Int, tWl::Float64, M::Array{Float64},
	P::Array{Float64}, Lambda::Float64,
	energies::Array{Float64}, rmax_prev::Int)

	H = zeros(Float64, rmax, rmax)

	# off-diagonal elements
	@inbounds for s in 1:2
		H += tWl * kron(view(M,s,:,:), view(P,s,:,:)) 
		# i think (-1)^k is unused if kron product is used
	end
	H += H' # Hermitian conjugate

	# diagonal elements
	H += sqrt(Lambda) * kron(I(4), diagm(energies[1:rmax_prev]))

	return H
end

	
# index slice for use in P matrix
function slice(k::Int64, r::Int64)
	return r*(k-1)+1:r*k
end # ie 1:r, r+1:2r, 2r+1:3r, 3r+1:4r

# compute P for next iteration
function update_P(rmax::Int, M::Array{Float64}, H::Array{Float64},
	rmax_prev::Int)
	P = zeros(Float64, 2, rmax, rmax)

	# we need to access slices of P, due to the 
	# definition of |N+1;r>_d in terms of
	# |N+1;k',r'>_bs; 4 slices equaly spaced from 1 to rmax_prev

	@threads for s in 1:2 # spins
	@inbounds for k in 1:4
	@inbounds for kp in 1:4
		if M[s,kp,k] == 0
			continue
		end
		
		# P(s,r,r') = <r| c_s^dagger |r'>

		P[s,:,:] += M[s,kp,k] * view(H,slice(k,rmax_prev),1:rmax)' *
			view(H,slice(kp,rmax_prev),1:rmax)
	end # kp
	end # k
	end # s

	return P
end

	
# thermodynamical quantities such as the entropy
function compute_thermo(l::Int, Lambda::Float64, betabar::Float64,
		energies::Array{Float64})

	# each quantity is for an interation l
	Tl = sqrt(Lambda)^(2-l) / betabar # temperature
	print("T = $(Tl)\n")

	# partition function Z, entropy S:
	Zl = sum(exp.(-betabar * energies))
	Sl = sum(betabar * energies .*
		exp.(-betabar * energies))/Zl + log(Zl)

	return Tl, Sl, Zl
end

# adjusted temperature, entropy across iterations
function adjusted_thermo(T::Array{Float64}, S::Array{Float64},
				S_noimp::Array{Float64})
	Ta = zeros(Float64, length(T)-1)
	Sa, Sa_noimp = similar(Ta), similar(Ta)

	@threads  for i in 1:length(Ta)
		Ta[i] = sqrt(T[i]*T[i+1])
		Sa[i] = (S[i]+S[i+1])/2
		Sa_noimp[i] = (S_noimp[i]+S_noimp[i+1])/2
	end

	return Ta, Sa, Sa_noimp
end

# energy flow of lowest states, for moving average
function energyflow_average(energy_flow::Array{Float64})
	enflow_avg = zeros(Float64, size(energy_flow,1), size(energy_flow,2) -1)

	@threads for i in 1:size(enflow_avg,2)
		enflow_avg[:,i] = (energy_flow[:,i] + energy_flow[:,i+1])/2
	end

	return enflow_avg
end


# main NRG function
function nrg(eps::Float64, U::Float64, M::Array{Float64}, Lambda::Float64, 
		lmax::Int, rlim::Int, betabar::Float64, num_energy_flow::Int, 
		tW::Array{Float64})

	# STEP 2: Initialize the Hamiltonian: impurity
	rmax, energies, P = initialize_ham(eps, U, Lambda, M)


	######## NRG loop

	# entropy, partition function, temperature for each iteration
	S = zeros(Float64, lmax+1)
	Z, T = similar(S), similar(S)

	# look at the energy flow per iteration
	energy_flow = zeros(Float64, num_energy_flow, length(S))

	@inbounds for l in 1:length(S)
		print("iteration $(l-1): ")

		rmax_prev::Int = rmax # dimension of diagonalised states,
			# at the previous iteration
		rmax *= 4 # dimension of Hilbert space at this iteration, 
			# grows by a factor of 4


		# STEP 3: new Hamiltonian
		H = new_ham(rmax, tW[l], M, P, Lambda, energies, rmax_prev)


		# STEP 4: diagonalize the Hamiltonian
		energies, H = eigen(H) # H becomes its eigenvector matrix

		# shift energies relative to the ground state for numerical stability
		energies .-= energies[1]

 		# energy flow, for plotting
		energy_flow[:,l] = energies[2:num_energy_flow+1]


		# STEP 5: compute thermodynamics, such as the entropy
		T[l], S[l], Z[l] = compute_thermo(l, Lambda, betabar, energies)


		# STEP 6: truncate the states, prepare for next iteration
		if l == length(S) # if last iteration, exit
			break
		end

		rmax = min(rmax, rlim) # truncation

		# STEP 7: update P matrix for next iteration
		P = update_P(rmax, M, H, rmax_prev)

	end # end of nrg loop

	return T, S, energy_flow
end # end of nrg function


	# main
let

# NRG parameters
Lambda::Float64 = 2.5 # discretization parameter
lmax::Int = 50 # number of iterations
rlim::Int = 600 # number of states to keep
betabar::Float64 = 1.1 # temperature parameter

# AIM parameters
U_imp::Float64 = 0.3
eps_imp::Float64 = -U_imp/2 # particle hole symmetric point value
V::Float64 = 0.1

# for no imp calculation, get S_imp
eps_noimp::Float64 = 20. 
U_noimp::Float64 = U_imp

# extra parameters
num_energy_flow::Int = 15 # number of energy flows to plot


# STEP 1: initialize the Wilson chain and coefficient matrix

M = init_coeff_matrix()
tW = init_wilson_chain(Lambda, lmax, V)


##### compute NRG for the whole system and the leads, to isolate impurity 


# whole system:
println("full system calculation:")
T, S, energy_flow = nrg(eps_imp, U_imp, M, Lambda, lmax, rlim,
			      betabar, num_energy_flow, tW)

hdfile = h5open("data.h5", "cw")

# energy flow plot
save_data(hdfile, "energy_flow", energy_flow)

p = plot(1:2:lmax+1, (energy_flow')[1:2:end, :], # avoid even/odd oscillations
		 xlabel="iteration", ylabel="energy", title="Energy flow",
	 legend=false, dpi=300)
savefig(p, "energy_flow.png")



# no imp:
println("\nno impurity calculation:")
T, S_noimp, energy_flow = nrg(eps_noimp, U_noimp, M, Lambda, lmax, rlim,
			      betabar, num_energy_flow, tW)

# entropy plot
Ta, Sa, Sa_noimp = adjusted_thermo(T, S, S_noimp)

save_data(hdfile, "T", Ta)
save_data(hdfile, "S", Sa .- Sa_noimp)

p = plot(Ta, (Sa .- Sa_noimp)/log(2), xscale=:log10, xlabel="T",
	ylabel="S_imp/ln2",
	 title="Impurity entropy", legend=false, dpi=300)
savefig(p, "entropy.png")



close(hdfile)

end
