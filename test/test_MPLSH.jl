using Test
#using MPLSH
include("../src/MPLSH.jl")

@testset "LSH tests" begin
    lsh_base = LSHBase(2, 1., 5)

    data = zeros(10, 2)
    data[:, 1] = vcat(randn(5) .+ 10, randn(5))
    data[:, 2] = vcat(randn(5) .+ 5, randn(5) .+ -2) 

    lsh = LSH(lsh_base, data)
    query = [9., 4.]
    nn_data = nearest_neighbors(lsh, query, 3, 10)
    @test length(nn_data) >= 3

    hashed = hash(lsh, data)
end;
