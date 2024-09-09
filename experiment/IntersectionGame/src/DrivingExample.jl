module DrivingExample

using MCPGameSolver: MCPGameSolver, MCPCoupledOptimizationSolver, WarmStartRecedingHorizonStrategy
using PATHSolver: PATHSolver
using TrajectoryGamesBase: TrajectoryGamesBase, TrajectoryGame, rollout,
  GeneralSumCostStructure, ProductDynamics, TrajectoryGame, TimeSeparableTrajectoryGameCost, TrajectoryGameCost,
  state_dim, control_dim, num_players, get_constraints, state_bounds, control_bounds
using LiftedTrajectoryGames: LiftedTrajectoryStrategy
using TrajectoryGamesExamples: TrajectoryGamesExamples, animate_sim_steps, planar_double_integrator, UnicycleDynamics,
  BicycleDynamics, create_environment_axis, time_invariant_linear_dynamics
using DifferentiableTrajectoryOptimization: DifferentiableTrajectoryOptimization, ParametricTrajectoryOptimizationProblem,
  MCPSolver, Optimizer, get_constraints_from_box_bounds
using ProgressMeter: ProgressMeter
using ParametricMCPs: ParametricMCPs
using IfElse: IfElse, ifelse
using BlockArrays: BlockArrays, Block, BlockVector, mortar, blocksizes, blocksize, blocks
using LazySets: LazySets
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using Random: Random, shuffle!
using Distributions: Distributions, Uniform, MixtureModel, Normal, MvNormal
using PDMats: PDiagMat
using Flux: Flux, gradient, Optimise.update!, params, glorot_uniform, Dense, Optimise.Adam, NNlib.relu, NNlib.gelu, NNlib.elu, 
        Chain, @functor, train!, cpu, gpu, softplus, params, BatchNorm, normalise
using LinearAlgebra: LinearAlgebra, norm_sqr, norm
using Statistics: mean, std
using StatsBase: sem, sample, Histogram, fit
using MCPGameSolver.ExampleProblems: n_player_collision_avoidance, two_player_guidance_game,
  two_player_guidance_game_with_collision_avoidance, shared_collision_avoidance_coupling_constraints
using Parameters
using GeometryBasics
using Plots: Plots, plot, scatter
using JLD2
# using GLMakie: GLMakie
using Makie: Makie, Axis, lines!, band!, lines, band, Legend, violin!, heatmap!
using FileIO
using Colors: @colorant_str, Colorant
using CSV: CSV
using DataFrames: DataFrames
using HypothesisTests: HypothesisTests
using Dates: now
using Clustering: Clustering, kmeans
using ContingencyGames: ContingencyGames, ContingencyGame, setup_contingency_game_solver, solve_contingency_game
using CairoMakie

include("train_vae.jl")
include("vae_games_utils.jl")
include("general_utils.jl")
include("game_definition/games.jl")
include("baselines/mle.jl")
include("simulation_script/intersection_inference.jl")
include("traffic_infrastructure/infrastructure.jl")
include("traffic_infrastructure/highway_roadway.jl")
include("traffic_infrastructure/merging_roadway.jl")
include("traffic_infrastructure/intersection_roadway.jl")

end # module
