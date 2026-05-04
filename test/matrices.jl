using Random: Xoshiro
using SparseArrays: sprand

const MATRICES = let rng = Xoshiro(0)
    [
        [1e6 0.0 3.0; 2.0 1e-3 0.0; 0.0 0.0 0.0; -4e2 5.0 7e-5],   # tall, ill-scaled, zero row
        [4.0 1e-3 0.0; 1e-3 1e6 2.0; 0.0 2.0 7e-2],                # square, ill-scaled
        [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0],                        # wide, well-conditioned
        [1.0 0.0 3.0; 2.0 4.0 0.0; -1.5 5.0 0.7],                  # square, well-conditioned
        [3.0 -1.0 0.0 2.0; -1.0 4.0 -2.0 0.0; 0.0 -2.0 5.0 -3.0],  # wide, mixed signs, banded zeros
        rand(rng, 30, 30) .* exp.(8 .* randn(rng, 30, 30)),        # square dense, ill-scaled
        rand(rng, 50, 80),                                         # wide dense, well-conditioned
        sprand(rng, 200, 120, 0.05),                               # rectangular sparse
        sprand(rng, 100, 100, 0.1),                                # square sparse
    ]
end
