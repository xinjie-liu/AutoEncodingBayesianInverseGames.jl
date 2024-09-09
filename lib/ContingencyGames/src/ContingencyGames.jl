module ContingencyGames
__precompile__(false)

# This file contains code from:
# Copyright (c) 2022 lassepe <lasse.peters@mailbox.org>
# Repository link: https://github.com/lassepe/ContingencyGames.jl

using TrajectoryGamesBase: TrajectoryGamesBase
using Makie: Makie
using BlockArrays: mortar

include("api.jl")

include("parameterized_dynamics.jl")

# Submodules
include("Solver/Solver.jl")
end
