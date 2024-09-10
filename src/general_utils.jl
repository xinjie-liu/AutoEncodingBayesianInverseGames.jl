#===========================================

This file contains general utility functions

============================================#


#======== environment =========#

struct RoadwayEnvironment{T}
    set::T
    roadway::Roadway
    lane_id_per_player::Vector
    radius::Float64
end

function RoadwayEnvironment(vertices::AbstractVector{<:AbstractVector{<:Real}}, roadway::Roadway, 
        lane_id_per_player::Vector, radius::Float64)
    RoadwayEnvironment(LazySets.VPolytope(vertices), roadway, lane_id_per_player, radius)
end

function plot_wall_constraint() end

function construct_intersection_roadway(; ll, lw, turn_radius)

    x1 = [ -ll/3, lw/2]
	x2 = [ ll/3, lw/2]
	x3 = [ -ll/3, -lw/2]
	x4 = [ ll/3, -lw/2]
	x5 = [ -lw/2, ll/2]
	x6 = [ lw/2, ll/2]
	x7 = [ -lw/2, -ll/2]
	x8 = [ lw/2, -ll/2]
    x9 = [ -lw/2, lw/2]
	x10 = [ lw/2, lw/2]
	x11 = [ -lw/2, -lw/2]
	x12 = [ lw/2, -lw/2]
    
    vertices = Vector([x1, x9, x5, x6, x10, x2, x4, x12, x8, x7, x11, x3]) # intersection
    # vertices = Vector([x5, x6, x10, x2, x4, x12, x8, x7]) # T-junction
    roadway_opts = FourIntersectionRoadwayOptions(; lane_length = ll, lane_width = lw, turn_radius)
    roadway = build_roadway(roadway_opts)

    vertices, roadway
end

function construct_env(num_player, ego_agent_id, vertices, roadway, collision_radius; ramp_merging = true)
    player_lane_id = 3 * ones(Int, num_player)
    if ramp_merging
        player_lane_id[ego_agent_id] = 4
    end
    environment = RoadwayEnvironment(vertices, roadway, player_lane_id, collision_radius)

    environment
end

function TrajectoryGamesBase.get_constraints(env::RoadwayEnvironment, player_idx)
    lane = env.roadway.lane[env.lane_id_per_player[player_idx]]
    walls = deepcopy(lane.wall)
    ri = env.radius
    for j = 1:length(walls)
        walls[j].p1 -= ri * walls[j].v
        walls[j].p2 -= ri * walls[j].v
    end
    function (state)
        position = state[1:2]
        # wall constraints
        wall_constraints = mapreduce(vcat, walls) do wall
            # finite-length wall constraint
            # left = max((position - wall.p1)' * (wall.p2 - wall.p1), 0)
            # right = max((position - wall.p2)' * (wall.p1 - wall.p2), 0)
            # product = (wall.p1 - position)' * wall.v
            # left * right * product

            # half-space constraint
            product = (wall.p1 - position)' * wall.v
        end
        # circle constraints
        if length(lane.circle) > 0
            # general circlar constraint
            # circle_constraints = mapreduce(vcat, lane.circle) do circle
            #     (position[1] - circle.x)^2 + (position[2] - circle.y)^2 - (circle.r + ri)^2
            # end

            # smoothened union of wall constraint
            # warning: this implementation is currently specialized for lane 13 and is not generalized for other lanes yet!
            products = map(2:3) do ii
                wall = walls[ii]
                product = (wall.p1 - position)' * wall.v
            end
            circle_constraints = [smooth_max(products[1], products[2]; sharpness = 5.0)]
        else
            circle_constraints = []
        end
        length(circle_constraints) > 0 ? circle_constraints : vcat(wall_constraints, circle_constraints)
    end
end

function smooth_max(vals...; sharpness = 3.0)
    # For improved numerical stability, we subtract the mean of the values
    c = mean(v * sharpness for v in vals)
    (1 / sharpness) * (c + log(sum(exp(sharpness * v - c) for v in vals)))
end

function construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
    collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ, game_type = "highway")
    dynamics = BicycleDynamics(; 
        l = 0.1,
        state_bounds = (; lb = [-Inf, -Inf, -max_velocity, -Inf], ub = [Inf, Inf, max_velocity, Inf]),
        control_bounds = (; lb = [-max_acceleration, -max_ϕ], ub = [max_acceleration, max_ϕ]),
        integration_scheme = :reverse_euler
    )
    if game_type == "highway"
        game = highway_game(num_player; min_distance, hard_constraints, collision_avoidance_coefficient,
            environment, dynamics)
    elseif game_type == "intersection"
        game = intersection_game(num_player; min_distance, hard_constraints, collision_avoidance_coefficient,
            environment, dynamics)
    end
    game
end

#======== visualization =========#

function visualize_button!(figure, label)
    button = Makie.Button(figure; label, tellwidth = false)
    clicked = Makie.Observable(false)
    Makie.on(button.clicks) do n
        clicked[] = true
    end
    (; button, clicked)
end

function visualize_prediction(predicted_opponents_trajectory; visualization, ego_agent_id, rng = nothing)
    # strategy_ = deepcopy(strategy.substrategies)
    # deleteat!(strategy_, ego_agent_id)
    predicted_strategy_visualization = map(1:length(predicted_opponents_trajectory)) do ii
        substrategy = Makie.Observable(predicted_opponents_trajectory[ii])
        TrajectoryGamesBase.visualize!(visualization.environment_axis, substrategy; color = colorant"rgba(238, 29, 37, 1.0)")
        substrategy
    end
end

function visualize_prediction(predicted_opponents_strategy::ContingencyGames.ContingencyStrategy; visualization, ego_agent_id, rng = nothing)
    branch_strategies = predicted_opponents_strategy.fields.branch_strategies
    weights = predicted_opponents_strategy.fields.weights
    predicted_strategy_visualization = map(1:length(branch_strategies)) do ii
        substrategy = contingency2liftedstrategy(branch_strategies[ii], ii, rng)
        substrategy = Makie.Observable(substrategy)
        transparency = weights[ii]
        TrajectoryGamesBase.visualize!(visualization.environment_axis, substrategy; color = parse(Colorant, "rgba(238, 29, 37, $transparency)"))
        substrategy
    end
end

function visualize_players!(
    axis,
    players;
    ego_agent_id,
    opponents_id,
    marker = '➤',
    transparency = 1.0,
    markersize = 20, 
) 
    for player_i in 1:blocksize(players[], 1)
        player_color = player_i == ego_agent_id ? parse(Colorant, "rgba(238, 29, 37, $transparency)") : parse(Colorant, "rgba(46,139,87, $transparency)")
        position = Makie.@lift Makie.Point2f($players[Block(player_i)][1:2])
        rotation = Makie.@lift $players[Block(player_i)][4]
        Makie.scatter!(axis, position; rotation, color = player_color, marker = load("media/car"*string(player_i)*".png"), markersize = 35)
    end
end

function visualize_history!(axis, observation; marker = '*', markersize = 15)
    for jj in 1:(observation[][1] |> blocks |> length)
        for ii in 1:length(observation[])
            position = Makie.@lift Makie.Point2f($observation[ii][Block(jj)][1:2])
            Makie.scatter!(axis, position; color = colorant"orange", marker, markersize)
        end
    end
end

function visualize_players_without_vel_digit!(
    axis,
    players;
    ego_agent_id,
    opponents_id,
    marker = '➤', 
    markersize = 20,
)
    for player_i in 1:blocksize(players[], 1)
        player_color = player_i == ego_agent_id ? colorant"yellow" : colorant"blue"
        position = Makie.@lift Makie.Point2f($players[Block(player_i)][1:2])
        rotation = Makie.@lift $players[Block(player_i)][3]
        Makie.scatter!(axis, position; rotation, color = player_color, marker, markersize)
    end
end

function visualize_obstacle_bounds!(
    axis,
    players;
    obstacle_radius = 1.0,
    ego_agent_id,
    opponents_id
)
    for player_i in 1:blocksize(players[], 1)
        player_color = player_i == ego_agent_id ? colorant"rgba(238, 29, 37, 1.0)" : colorant"rgba(46,139,87, 1.0)"
        position = Makie.@lift Makie.Point2f($players[Block(player_i)][1:2])
        Makie.scatter!(
            axis,
            position;
            color = (player_color, 0.4),
            markersize = 2 * obstacle_radius,
            markerspace = :data,
        )
    end
end

function TrajectoryGamesBase.visualize!(
    canvas,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy};
    color = :black,
    weight_offset = 0.0,
    markersize_range = [8, 16],
)
    horizon = γ[].trajectories[1].xs |> length
    Δ = (markersize_range[2]-markersize_range[1]) / (horizon-1)
    for ii in 1:horizon
        pos = Makie.@lift Makie.Point2f($γ.trajectories[1].xs[ii][1:2])
        rotation = Makie.@lift $γ.trajectories[1].xs[ii][4]
        Makie.scatter!(canvas, pos; rotation, color, markersize = markersize_range[1]+Δ*(ii-1))
    end
    # trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    # Makie.series!(canvas, γ; color = trajectory_colors, linewidth=3.5)
end

function visualize!(figure, game, pointmasses, strategy; targets = nothing, obstacle_radius = 0.0, 
    ego_agent_id = nothing, opponents_id = nothing)

    ll = game.env.roadway.opts.lane_length
    lw = game.env.roadway.opts.lane_width

    num_player = length(strategy.substrategies)
    player_colors = map(1:num_player) do ii
        color = ii == ego_agent_id ? colorant"rgba(238, 29, 37, 1.0)" : colorant"rgba(46,139,87,1.0)"
    end

    environment_axis = create_environment_axis(figure[1, 1], game.env; title = "Game")
    Makie.hidedecorations!(environment_axis)

    pointmasses = Makie.Observable(pointmasses)
    visualize_players!(environment_axis, pointmasses; ego_agent_id, opponents_id)
    if obstacle_radius > 0
        visualize_obstacle_bounds!(environment_axis, pointmasses; 
        obstacle_radius, ego_agent_id, opponents_id)
    end

    strategy = Makie.Observable(strategy)
    TrajectoryGamesBase.visualize!(environment_axis, strategy; colors = player_colors)
    if !isnothing(targets)
        targets = Makie.Observable(targets)
        visualize_targets!(environment_axis, targets)
    end

    skip_button = visualize_button!(figure, "Skip")
    stop_button = visualize_button!(figure, "Stop")
    pause_button = visualize_button!(figure, "Pause")
    continue_button = visualize_button!(figure, "Continue")
    button_grid = Makie.GridLayout(tellwidth = false)
    button_grid[1, 1] = skip_button.button
    button_grid[1, 2] = stop_button.button
    button_grid[1, 3] = pause_button.button
    button_grid[1, 4] = continue_button.button
    figure[2, 1] = button_grid

    Makie.xlims!(environment_axis, -0.6 * ll, 0.6 * ll)
    Makie.ylims!(environment_axis, -0.6 * ll, 0.6 * ll) 

    (; pointmasses, strategy, targets, environment_axis, skip_button, stop_button, pause_button, continue_button)
end

mutable struct ExtraVisualization
    history::Any
    goal_estimates::Any
    VAE_samples::Any
end

function visualize_mle_estimates!(axis, estimate; t, n_sim_steps)
    markersize_range = [4, 25]
    Δ = (markersize_range[2]-markersize_range[1])/(n_sim_steps)
    markersize = markersize_range[1] + Δ*t
    Makie.scatter!(axis, estimate[Block(2)][1], estimate[Block(2)][2]; marker = :star4, markersize, color = colorant"green")
end

function visualize_targets!(
    axis,
    targets;
    player_colors = range(colorant"red", colorant"green", length = blocksize(targets[], 1)),
    marker = :diamond,
)
    for player_i in 1:blocksize(targets[], 1)
        color = player_colors[player_i]
        target = Makie.@lift Makie.Point2f($targets[Block(player_i)])
        Makie.scatter!(axis, target; color, marker, markersize = 20)
    end
end

function TrajectoryGamesBase.visualize!(canvas, environment::RoadwayEnvironment; color = colorant"rgba(225, 242, 251, 1.0)") #(19, 45, 82, 1.0)")
    geometry = GeometryBasics.Polygon(GeometryBasics.Point{2}.(environment.set.vertices))
    Makie.poly!(canvas, geometry; color)

	ll = environment.roadway.opts.lane_length
	lw = environment.roadway.opts.lane_width
    
    x1 = [ -ll/3, lw/2]
	x2 = [ ll/3, lw/2]
	x3 = [ -ll/3, -lw/2]
	x4 = [ ll/3, -lw/2]
	x5 = [ -lw/2, ll/2]
	x6 = [ lw/2, ll/2]
	x7 = [ -lw/2, -ll/2]
	x8 = [ lw/2, -ll/2]
	x9 = [ -lw/2, lw/2]
	x10 = [ lw/2, lw/2]
	x11 = [ -lw/2, -lw/2]
	x12 = [ lw/2, -lw/2]
	x13 = [ -ll/3, 0.]
	x14 = [ -lw/2, 0.]
	x15 = [ lw/2, 0.]
	x16 = [ ll/3, 0.]
	x17 = [ 0., ll/2]
	x18 = [ 0., lw/2]
	x19 = [ 0., -lw/2]
	x20 = [ 0., -ll/2]

    Makie.lines!(canvas, [x5[1], x9[1]], [x5[2], x9[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x11[1], x7[1]], [x11[2], x7[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x1[1], x9[1] .+ 0.006], [x1[2], x9[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x3[1], x11[1] .+ 0.006], [x3[2], x11[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x6[1], x10[1]], [x6[2], x10[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x12[1], x8[1]], [x12[2], x8[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x10[1] .- 0.006, x2[1]], [x10[2], x2[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x12[1] .- 0.006, x4[1]], [x12[2], x4[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(canvas, [x17[1], x18[1]], [x17[2], x18[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(canvas, [x13[1], x14[1]], [x13[2], x14[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(canvas, [x15[1], x16[1]], [x15[2], x16[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(canvas, [x19[1] .- 0.006, x20[1]], [x19[2], x20[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
end

#================== other ================#

function sample_random_initial_guess(; rng, lw, collision_radius, ego_agent_id, goal, num_player)
    x = rand(rng, Distributions.Uniform(-(lw/2-collision_radius), (lw/2)*2))
    y = rand(rng, Distributions.Uniform(-(lw/2)*2, -collision_radius))
    random_goal = mortar([[x, y] for ii in 1:num_player])
    random_goal[Block(ego_agent_id)] = goal[Block(ego_agent_id)] # ego goal known
    random_goal
end

function solve_game_with_resolve!(receding_horizon_strategy, game, system_state)
    """
    solve forward game, resolve with constant velocity rollout as initialization if solve failed
    """
    strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
        game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
    if receding_horizon_strategy.solution_status != PATHSolver.MCP_Solved
        @info "Solve failed, re-initializing..."
        receding_horizon_strategy.last_solution = nothing
        receding_horizon_strategy.solution_status = nothing
        strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, 
            game, system_state, receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
    end

    strategy
end

function check_solver_status!(
    receding_horizon_strategy_ego, strategy, 
    strategy_ego, game, system_state, ego_agent_id, horizon, max_acceleration, rng
)
    """
    Check solver status, if failed, overwrite with an emergency strategy
    """
    solving_status = receding_horizon_strategy_ego.solution_status
    if solving_status == PATHSolver.MCP_Solved
        strategy.substrategies[ego_agent_id] = strategy_ego.substrategies[ego_agent_id]
    else
        dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
            control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
            max_acceleration = max_acceleration, strategy_type = "max_acceleration")
        strategy.substrategies[ego_agent_id] = dummy_substrategy
    end

    solving_status
end

function construct_observation_index_set(;
    num_player, ego_agent_id, vector_size, state_dimension, mcp_game,
)
    observation_idx_set = mapreduce(vcat, 1:num_player) do ii
        # if ii != ego_agent_id
        index = []
        for jj in 1:vector_size
            offset = state_dimension * (jj - 1)
            # partial observation
            index = vcat(index, mcp_game.index_sets.τ_idx_set[ii][[offset + 1, offset + 2, offset + 4]])
        end     
        # else
        #     index = []
        # end
        # # full observation
        # index = ii != ego_agent_id ? mcp_game.index_sets.τ_idx_set[ii][1:(vector_size * state_dimension)] : []
        index
    end
    sort!(observation_idx_set)

    observation_idx_set
end

function erase_last_solution!(receding_horizon_strategy)
    # clean up the last solution
    receding_horizon_strategy.last_solution = nothing
    receding_horizon_strategy.solution_status = nothing
end

function create_dummy_strategy(game, system_state, control_dimension, horizon, player_id, rng;
    max_acceleration = nothing, strategy_type = "zero_input")
    @assert strategy_type in ["zero_input", "max_acceleration"] "Please give a valid strategy type."
    if strategy_type == "zero_input"
    dummy_strategy = (x, t) -> zeros(control_dimension)
    else
        dummy_strategy = let
            function max_acc_strategy(x, t)
                control = zeros(control_dimension)
                if x[3] >= 0
                    control[1] = -max_acceleration
                elseif x[3] < 0
                    control[1] = max_acceleration
                end
                # double integrator dynamics
                # if x[4] >= 0
                #     control[2] = -max_acceleration
                # end
                control
            end
        end
    end

    dummy_xs = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).xs
    dummy_us = rollout(game.dynamics.subsystems[player_id], 
        dummy_strategy, system_state[Block(player_id)], horizon + 1).us
    dummy_trajectory = (; xs = dummy_xs, us = dummy_us)
    
    (; dummy_substrategy = LiftedTrajectoryStrategy(player_id, [dummy_trajectory], [1], nothing, rng, Ref(0)), 
        dummy_trajectory)
end

function construct_intersection_dataset(game, horizon, rng, num_player, ego_agent_id, collision_radius, number_trials; 
    max_velocity = 0.5, vertices)
    println("Sampling initial states and goals...")
    ll = game.env.roadway.opts.lane_length
    lw = game.env.roadway.opts.lane_width
    turn_radius = game.env.roadway.opts.turn_radius
    initial_state_set = map(1:number_trials) do ii
        mortar([
            [lw/4, rand(rng, Distributions.Uniform(-ll/1.94, -ll/2.55)), 0.0, π/2], # ego
            [-collision_radius, ll/2.5, 0.0, -π/2]]) # opponent
    end
    normal_distribution_left_turn = MvNormal([1.5*lw/2, -0.5*lw/2], PDiagMat([((lw/2)/16)^2, ((lw/2-2*collision_radius)/6)^2]))
    normal_distribution_straight = MvNormal([-0.5*lw/2, -1.5*lw/2], PDiagMat([((lw/2-2*collision_radius)/6)^2, ((lw/2)/16)^2]))
    opponent_goal_distribution = MixtureModel(MvNormal[normal_distribution_left_turn, normal_distribution_straight])
    opponent_goal_samples = rand(rng, opponent_goal_distribution, number_trials)
    xs = clamp.(opponent_goal_samples[1, :], -(lw/2 - collision_radius), Inf)
    ys = clamp.(opponent_goal_samples[2, :], -Inf, -collision_radius)

    goal_dataset = map(1:number_trials) do ii
        # a uniform distribution
        # x = rand(rng, Distributions.Uniform(-(lw/2 - collision_radius), ll/4))
        # y = rand(rng, Distributions.Uniform(-ll/4, -collision_radius))
        
        # sampled goals from the Gaussian mixture
        x = xs[ii]
        y = ys[ii]
        mortar([[lw/4, ll/4], [x, y]])
    end

    initial_state_set, goal_dataset, opponent_goal_distribution
end

function compute_distance(system_state)
    num_player = blocksizes(system_state) |> only |> length
    distance = mapreduce(vcat, 1:(num_player - 1)) do player_i
        mapreduce(vcat, (player_i + 1):num_player) do paired_player
            norm(system_state[Block(player_i)][1:2] - system_state[Block(paired_player)][1:2])
        end
    end 
end

function initialize_result_data(; solver_string_lst)
    result_data = Dict()
    for solver in solver_string_lst
        result_data[solver] = []
    end
    result_data
end

function initialize_episode_data()
    episode_data = (; beliefs = (; weights = [], supports = []), 
        states = [], true_goal = [], predicted_opponents_trajectory = [], 
        goal_estimation = [], controls = [], 
        solving_time = [], infeasible_solve = [])
end

function load_episode_data!(; episode_data, solver_string, system_state, control, goal, time_exec, 
    goal_estimation, predicted_opponents_trajectory, belief, solve_status)
    push!(episode_data.states, system_state)
    push!(episode_data.controls, control)
    push!(episode_data.true_goal, goal)
    if !isnothing(predicted_opponents_trajectory)
        push!(episode_data.predicted_opponents_trajectory, predicted_opponents_trajectory)
    end
    if !isnothing(time_exec)
        push!(episode_data.solving_time, time_exec)
    end
    if solver_string == "B-PinE"
        push!(episode_data.beliefs.weights, [belief[1].weight, belief[2].weight])
        push!(episode_data.beliefs.supports, [belief[1].cost_parameters, belief[2].cost_parameters])
        infeasible_solve = solve_status == PATHSolver.MCP_Solved ? 0 : 1
        push!(episode_data.infeasible_solve, infeasible_solve)
    else
        push!(episode_data.goal_estimation, goal_estimation)
    end
end

function compute_state_estimation(previous_state, system_state, num_player; dt = 0.1)
    estimated_state = deepcopy(previous_state)
    for ii in 1:num_player
        estimated_velocity = ()
        estimated_state[Block(ii)][3:4] = (system_state[Block(ii)][1:2] - previous_state[Block(ii)][1:2]) / dt
    end
    estimated_state
end

function update_observation_vec!(; solver_string, xs_observation, xs_pre, system_state, num_player, ego_agent_id, previous_state, vector_size)
    if solver_string != "B-PinE"
        push!(xs_observation, reduce(vcat, [system_state[Block(ii)][[1, 2, 4]] for ii in 1:num_player])) # partial observation
        # push!(xs_observation, reduce(vcat, [system_state[Block(ii)] for ii in 1:num_player if ii != ego_agent_id])) # full observation
        estimated_state = previous_state # compute_state_estimation(previous_state, system_state, num_player)
        push!(xs_pre, estimated_state)
        if length(xs_observation) > vector_size
            popfirst!(xs_observation)
            popfirst!(xs_pre)
        end
    elseif solver_string == "B-PinE"
        ego_agent_id == 1 || throw(ArgumentError("Ego agent must have an index 1."))
        if length(xs_observation) == 0
            for ii in 1:num_player
                push!(xs_observation, Vector{Any}[])
            end
        end
        for ii in 1:num_player
            if ii == ego_agent_id
                push!(xs_observation[ii], [system_state[Block(ii)][2]])
            else
                push!(xs_observation[ii], system_state[Block(ii)][[1, 2, 4]])     
            end
        end
        if length(xs_observation[1]) > vector_size
            for ii in 1:num_player
                popfirst!(xs_observation[ii])
            end
        end
    end
end  

function extract_observation(xs_observation; ego_agent_id, num_player, observation_dim = 3)
    xs_observation[1] |> length == num_player * observation_dim || throw(ArgumentError("Wrong observation vector dimension."))
    ego_agent_id == 1 || throw(ArgumentError("Ego agent must have an index 1."))
    map(1:num_player) do ii
        if ii == ego_agent_id
            obs = map(1:length(xs_observation)) do tt
                xs_observation[tt][2]
            end
        else
            obs = map(1:length(xs_observation)) do tt
                offset = (ii-1)*observation_dim
                xs_observation[tt][offset+1:offset+observation_dim]
            end
        end
        obs
    end
end

function my_norm_sqr(x)
    x'*x
end


function my_norm(x; regularization= 1e-4)
    sqrt(sum(x' * x) + regularization)
end

#============ Planning (forward game) ================#

function contingency2liftedstrategy(contingency_strategy, strategy_id, rng = nothing)
    LiftedTrajectoryStrategy(strategy_id, 
        [(; contingency_strategy.xs, contingency_strategy.us)], [1], nothing, rng, Ref(0))
end

function solve_contingency_game_with_warm_start(; contingency_game_solver, contingency_game, initial_belief, last_solution)          
    if !isnothing(last_solution)
        initial_guess = last_solution.status == PATHSolver.MCP_Solved ? last_solution.z : nothing
    else
        initial_guess = nothing
    end
    solve_contingency_game(contingency_game_solver, contingency_game, initial_belief; initial_guess)
end

function apply_braking!(; strategy, game, system_state, ego_agent_id, horizon, max_acceleration, rng)    
    braking_strategy, _ = create_dummy_strategy(game, system_state, 
    control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng;
    max_acceleration = max_acceleration, strategy_type = "max_acceleration")
    strategy.substrategies[ego_agent_id] = braking_strategy
end

function apply_ego_contingency_plan!(; strategy, strategy_ego, ego_agent_id, system_state, rng)
    strategy_ego.xs[1] = system_state[Block(ego_agent_id)] # hack for numerical deviations
    push!(strategy_ego.xs, strategy_ego.xs[end]) # a hack to fix length gap between two types of strategies
    converted_strategy_ego = contingency2liftedstrategy(strategy_ego, 1, rng)
    strategy.substrategies[ego_agent_id] = converted_strategy_ego
end



