# This file contains code from:
# Copyright (c) 2022 lassepe <lasse.peters@mailbox.org>
# Repository link: https://github.com/lassepe/ContingencyGames.jl

"""
    solve_contingency_game(
        solver,
        game::ContingencyGame,
        belief;
        initial_guess = nothing
    )

Solve a contingency `game` using `solver` with the given and `belief`. The belief is a
vector of named tuples of the form `(; weight, state, cost_parameters, dynamics_parameters)`.

Optionally, the user may provide an `initial_guess` for warm-starting the solver.
"""
function solve_contingency_game end

struct ContingencyGame{T}
    """
    (;
        horizon::Int
        parameterized_game::TrajectoryGame
    )
    """
    fields::T
end

"""
Generate a contingency game and planner from a vanilla game.
"""
function setup_contingency_game_solver(;
    game,
    initial_state,
    ground_truth_parameters,
    number_of_hypotheses=2,
    horizon=15,
    solver_configuration_kwargs=(;
        # fixing the branching time to reduce problem size
        branching_time_configuration=ContingencyGames.Solver.StaticBranchingTime(horizon),
    ),
    trigger_compilation=true,
    verbose=true,
)
    # TODO: @xinjie: As for the regular MCP solver, this involves a lot of compilation so you
    # may want to do this outside of this function or otherwise cache the results.
    contingency_game = ContingencyGame((; horizon, parameterized_game=game))
    dummy_parameters = zero(ground_truth_parameters)
    initial_belief =
        [
            (;
                weight=1 / number_of_hypotheses,
                state=initial_state,
                # Note: the exact parameter values don't matter here;
                # we just need the belief to trace it symbolically for compilation of the solver.
                cost_parameters=copy(dummy_parameters),
                dynamics_parameters=nothing,
            ) for _ in 1:number_of_hypotheses
        ]
    if verbose
        @info "Generating contingency game solver..."
    end
    contingency_game_solver = ContingencyGames.Solver.MCPContingencySolver(
        contingency_game,
        initial_belief;
        solver_configuration_kwargs...,
    )
    if trigger_compilation
        @info "Compiling contingency game solver..."
        ContingencyGames.solve_contingency_game(contingency_game_solver, contingency_game, initial_belief)
    end
    (; contingency_game_solver, contingency_game, initial_belief)
end

# TODO: annoying hack. probably want to have the contingency game have a collecion of costs rather
# than full games
function Base.getproperty(contingency_game::ContingencyGame, name::Symbol)
    if name === :env
        contingency_game.parameterized_game.env
    elseif name === :dynamics
        contingency_game.parameterized_game.dynamics
    elseif name === :cost
        contingency_game.parameterized_game.cost
    elseif name === :coupling_constraints
        contingency_game.parameterized_game.coupling_constraints
    else
        getproperty(getfield(contingency_game, :fields), name)
    end
end

function TrajectoryGamesBase.num_players(game::ContingencyGame)
    TrajectoryGamesBase.num_players(game.parameterized_game)
end
#=== Strategy ===#

struct ContingencyStrategy{T}
    """
    player_index::Integer,
    branching_time::Integer,
    branch_strategies::Vector
    weights::Vector{Float64},
    verbose::Bool = true,
    """
    fields::T
end

function (strategy::ContingencyStrategy)(state, time)
    # we chose the MAP branch. If the `time` is prior to `tb`, this doesn't make a difference because both branches are enforced to be identical.
    branch_index = argmax(strategy.weights)

    # evaluation of the contingency strategy beyond the branching_time is not meaningful because
    # we don't know which plan to apply
    # TODO: avoid hard-coding of certainty threshold
    if time > strategy.branching_time && strategy.weights[branch_index] < 0.99
        get(strategy.fields, :verbose, false) && @warn """
            Evaluating contingency strategy beyond branching time without sufficient certainty

            time: $(time)
            branching_time: $(strategy.branching_time)
            weights: $(strategy.weights)
            """
    end

    # the control input we want to apply
    u = strategy.branch_strategies[branch_index](state, time)

    # check that the other control input we *could* apply are not consistent
    other_branch_indices = setdiff(eachindex(strategy.branch_strategies), branch_index)
    if time <= strategy.branching_time && any(
        !isapprox(branch_strategy(state, time), u; atol = 1e-3) for
        branch_strategy in strategy.branch_strategies[other_branch_indices]
    )
        @warn """
        Inconsistent control inputs for player $(strategy.player_index). Candiates are:
        $(prod("\nu$i: " * string(branch_strategy(state, time)) for (i, branch_strategy) in enumerate(strategy.branch_strategies)))...
        """
    end

    u
end

function predict_state(strategy::ContingencyStrategy, time)
    @assert time <= strategy.branching_time
    branch_index = 1
    predict_state(strategy::ContingencyStrategy, time, branch_index)
end

function predict_state(strategy::ContingencyStrategy, time, branch_index)
    predict_state(strategy.branch_strategies[branch_index], time)
end

function Base.getproperty(solver::ContingencyStrategy, name::Symbol)
    if name === :fields
        Base.getfield(solver, name)
    else
        Base.getproperty(solver.fields, name)
    end
end

function predict_state(strategy::TrajectoryGamesBase.OpenLoopStrategy, time)
    strategy.xs[time]
end

# TODO: this is slight type piracy but fine for a rapid prototype
# Should be moved to TrajectoryGamesBase maybe

function predict_state(strategy::TrajectoryGamesBase.JointStrategy, time, args...)
    mortar([predict_state(substrategy, time, args...) for substrategy in strategy.substrategies])
end
