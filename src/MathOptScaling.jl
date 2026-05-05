module MathOptScaling

import SparseArrays

include("utils.jl")
include("algorithms/ruiz.jl")
include("algorithms/symmetric_ruiz.jl")
include("algorithms/diagonal_norm.jl")
include("algorithms/chambolle_pock.jl")
include("algorithms/tomlin.jl")
include("algorithms/curtis_reid.jl")

end
