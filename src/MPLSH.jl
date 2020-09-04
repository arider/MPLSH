
import DataStructures: BinaryMinHeap
using LinearAlgebra: norm

struct LSHBase
    alphas::Matrix{Float64} # Columns are linear equations.
    betas::Array{Float64, 2} # One per hash / one per alpha column.
    w::Array{Float64, 2} # Normalizing constant per hash.
end

"""
A gaussian LSH initialization method.

args:
    - n_dims: the number of dimensions in the input data.
    - w: the denominator and the max adjust value. May be an array of numbers.
    - n_hashes: number of hashes.
"""
function LSHBase(n_dims::Int, w=1.0, n_hashes=0)
    if n_hashes == 0
        n_hashes = Int(max(round(sqrt(n_dims)), 2))
    end
    alphas = randn(n_dims, n_hashes)

    # Normalize the alphas.
    alpha_norms = reshape([norm(col, 2) for col in eachcol(alphas)], 1, size(alphas, 2))
    unit_alphas = alphas ./ alpha_norms
    alphas = unit_alphas
 
    # Make sure that w is a column vector or an int.
    if isa(w, Float64) || isa(w, Int)
        w = reshape([w for _ in 1:n_hashes], 1, n_hashes)
    else
        w = reshape(w, 1, n_hashes)
    end
    betas = rand(1, n_hashes) .* w

    LSHBase(alphas, betas, w)
end

"""
The hash function.
"""
function hash(lsh::LSHBase, data)
    if length(size(data)) == 1
        data = reshape(data, 1, maximum(size(data)))
        hash_address = map(floor, ((data * lsh.alphas) .+ lsh.betas) ./ lsh.w)
        return vec(map(Int, hash_address))
    else
        hash_address = map(floor, ((data * lsh.alphas) .+ lsh.betas) ./ lsh.w)
        try
            return map(Int, hash_address)
        catch
            throw(error(join(["EXCEPTION", hash_address, data, lsh.alphas, lsh.betas, lsh.w], ", ")))
        end
    end
end

"""
Function to aggregate raw data into hash buckets.
"""
function reservoir_function(lshb::LSHBase, data::Array{Float64, 2})
    # Update vals
    vals = Dict{Array{Int, 1}, Any}()
    hashed = hash(lshb, data)
    for i in 1:size(hashed, 1)
        if !haskey(vals, hashed[i, :])
            vals[hashed[i, :]] = Array{Any, 1}()
        end
        push!(vals[hashed[i, :]], data[i, :])
    end
    return vals
end

"""
Function to aggregate raw data into hash buckets as average and count lists.
"""
function mean_count_function(lshb::LSHBase, data::Array{Float64, 2})
    vals = Dict{Array{Int, 1}, Any}()
    hashed = hash(lshb, data)
    for i in 1:size(hashed, 1)
        if !haskey(vals, hashed[i, :])
            vals[hashed[i, :]] = Array{Union{Int, Array{Float64, 1}}, 1}([[0. for _ in data[1, :]], 0])
        end
        vals[hashed[i, :]][1] += data[i, :]
        vals[hashed[i, :]][2] += 1
    end
    return vals
end

mutable struct LSH
    lshb::LSHBase
    aggregate_function::Function
    # Hash address to value.
    vals::Dict{Array{Int, 1}, Any}
end

"""
Create a new randomized LSH
args:
    - data: raw input data.
    - aggregate_function: a function to aggregate raw data points that fall
      into the same hash bucket.
    - n_hashes: the number of hashes.

"""
function LSH(data::Array{Float64, 2}, aggregate_function::Function=reservoir_function, n_hashes=0, w=.1)
    best = LSHBase(size(data, 2), w, n_hashes)

    # Update vals
    vals = aggregate_function(best, data)
    return LSH(best, aggregate_function, vals)
end

"""
Create a new LSH using the values in the provided LSHBase.
"""
function LSH(lshb::LSHBase, data::Array{Float64, 2}, aggregate_function::Function=reservoir_function)
    vals = aggregate_function(lshb, data)
    return LSH(lshb, aggregate_function, vals)
end

function hash(lsh::LSH, v)
    return hash(lsh.lshb, v)
end

"""
Calculate the score of a probe as the sum over the distance of the point from
the hash boundary + delta.

args:
    - query: a hash address.
    - delta: array of ints to apply to the hash address to get the probe
             location.
"""
function probe_score(lsh::LSHBase, query, delta)
    hashed_float = (query' * lsh.alphas .+ lsh.betas) ./ lsh.w
    if delta < 0
        return (map(floor, hashed_float) - hashed_float .+ [delta for _ in lsh.w]).^2
    else
        return (map(ceil, hashed_float) - hashed_float .+ [delta for _ in lsh.w]).^2
    end
end

"""
Generate query deltas in order of their probe_score and evaluate the probes for
neighbors. Return the neighbor hash addresses.

args
    lsh: mplsh object
    query: vector of data in original space
    target_neighbors: how many neighbors to return
    n_probes: the max number of probes to try
    probe_distance: positive int how many buckets away from query to include
                    in search.
"""
function get_neighbor_addresses(lsh::LSH, query, target_neighbors=1, probe_distance=1, n_probes=typemax(Int))
    neighbor_addresses = []
    query_address = hash(lsh, query)

    # Compute the probe_scores for all nonzero integer deltas in
    # [-probe_distance, probe_distance] (for each pair (i, delta_i))
    # perturbation_set will be (score, variable_index, probe_distance)
    perturbation_set = []
    for pd in 1:probe_distance
        perturbation_set = vcat(perturbation_set, vec([(score, hash_index, pd) for (hash_index, score) in enumerate(probe_score(lsh.lshb, query, pd))]))
        perturbation_set = vcat(perturbation_set, vec([(score, hash_index, -pd) for (hash_index, score) in enumerate(probe_score(lsh.lshb, query, -pd))]))
    end

    # Sort the pairs by score.
    scores = [pair[1] for pair in perturbation_set]
    perturbation_set = perturbation_set[sortperm(scores)]

    # Initialize the heap. The value [1] indicates that we'll start by trying
    # the first element in perturbation_set, which is sorted by cost.
    h = BinaryMinHeap{Tuple{Float64, Array{Int, 1}}}()
    push!(h, (perturbation_set[1][1], [1]))
    
    deltas = []
    count = 0
    n_probes_so_far = 0
    while length(deltas) < n_probes && length(h) > 0
        n_probes_so_far += 1
        count += 1
        # Get the next best probe.
        ai = pop!(h)

        # Create the hash address for this probe.
        probe = copy(query_address)
        for ind in ai[2]
            probe[perturbation_set[ind][2]] += perturbation_set[ind][3]
        end

        # If the bucket this delta indicates is not empty, add it to the neighbors set.
        if haskey(lsh.vals, probe)
            push!(neighbor_addresses, probe)
        end

        # If we've got the desired number of actual nonempty neighbors, return them.
        if length(neighbor_addresses) >= target_neighbors
            return neighbor_addresses
        end

        push!(deltas, ai[2])

        # Shift
        set = copy(ai[2])
        score = ai[1]
        # Remove the score for the current end perturbation.
        score -= perturbation_set[ai[2][end]][1]
        # Add the score for the new end perturbation.
        next = ai[2][end] + 1
        if next <= length(perturbation_set)
            score += perturbation_set[next][1]
            set[end] += 1
            push!(h, (score, set))
        end
       
        # Expand
        set = copy(ai[2])
        set = vcat(set, set[end] + 1)

        # Already updated the set at this point, offsets reflect.
        if length(set) <= length(lsh.lshb.w) - 1 && set[end] <= length(perturbation_set)
            score = ai[1] + perturbation_set[set[end]][1]
            push!(h, (score, set))
        end
    end

    return neighbor_addresses
end

"""
Find the approximate nearest neighbors with some constraints.

use:
Retrieve the first non-empty neighboring bucket's contents that can be found
adjacent to the query location.
    nearest_neighbors(lsh, [.23, -1, 3])
Retrieve the first non-empty neighboring bucket's contents within +-100 buckets
of the query location.
    nearest_neighbors(lsh, [.23, -1, 3], 1, probe_distance=100)
Retrieve the first 10 non-empty neighboring bucket's contents within +-100 buckets
of the query location.
    nearest_neighbors(lsh, [.23, -1, 3], 10, probe_distance=100)
And so on....

args
    lsh: mplsh object
    query: vector of data in original space
    n_neighbors: how many neighbors to return
    n_probes: the max number of probes to try
    probe_distance: positive int how many buckets away from query to include
                    in search.
"""
function nearest_neighbors(lsh::LSH, query, n_neighbors=1, probe_distance=3, n_probes=typemax(Int))
    nns = get_neighbor_addresses(lsh, query, n_neighbors, probe_distance, n_probes)
    neighbors = []
    for nn in nns
        neighbors = vcat(neighbors, lsh.vals[nn])
    end
    return Array{Float64, 2}(hcat(neighbors...)')
end


