#============================================================================================ 

A simulation script that runs different approaches in the traffic intersection example

Simulation videos are stored in the "data/" folder

============================================================================================#

function run_intersection_inference(; 
    number_trials = 3, solver = nothing, num_player = 2, ego_agent_id = 1, 
    lane_id_per_player = [10, 13],
    ll = 2.0, lw = 0.45, turn_radius = 0.3, # lane length, width, turning radius
    collision_radius = 0.08, max_velocity = 0.2, max_acceleration = 0.12, max_ϕ = π/4,
    collision_avoidance_coefficient = 400, hard_constraints = false, # collision avoidance costs and inequalities
    rng = Random.MersenneTwister(26), horizon = 15, n_sim_steps = 76,
    vector_size = 15, # number of position points that the ego keeps as observation
    turn_length = 1, # number of steps to take along the MPGP horizon
    max_grad_steps = 10, # max online gradient steps for MLE baseline
    lr = 2.1e-2, # step size for graident descent of MLE baseline
    root_folder = "data/",
    save = false,
    training_dataset_size = 700, #argument for loading training data set to get normalization factors
    episode_slicing_interval = 1, 
)

    #=================================#
    # Construction of game objects
    #=================================#
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)
    vertices, roadway = construct_intersection_roadway(; ll, lw, turn_radius)
    environment = RoadwayEnvironment(vertices, roadway, lane_id_per_player, collision_radius)
    game = construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
        collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ, game_type = "intersection")
    # jldsave(root_folder*"game.jld2"; game)
    initial_state_set, goal_dataset, target_lane_distribution_opp = construct_intersection_dataset(game, horizon, 
        rng, num_player, ego_agent_id, collision_radius, number_trials; max_velocity, vertices)
    initial_state = initial_state_set[1]
    system_state = initial_state
    # get dimensions and index sets
    state_dimension = state_dim(game.dynamics.subsystems[1])
    control_dimension = control_dim(game.dynamics.subsystems[1])
    ego_state_idx = let
        offset = ego_agent_id != 1 ? sum([blocksizes(initial_state)[1][ii] for ii in 1:(ego_agent_id - 1)]) : 0
        Vector((offset + 1):(offset + blocksizes(initial_state)[1][ego_agent_id]))
    end
    # the approaches to run; options: "GT", "B-PinE" (ours), "B-MAP" (ours), "R-MLE" (liu2023ral), "BP-MLE", "St-BP"
    solver_string_lst = ["GT", "B-PinE", "B-MAP", "R-MLE", "BP-MLE", "St-BP"]
    solver = @something(solver, MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal_dataset[1], 1))) # public solver for the uncontrolled agents
    mcp_game = solver.mcp_game

    #====================================#
    # Initialization of the solvers
    #====================================#
    observation_idx_set = construct_observation_index_set(;
        num_player, ego_agent_id, vector_size, state_dimension, mcp_game,
    )
    block_sizes_params = blocksizes(goal_dataset[1]) |> only
    # B-PinE
    vae, set_up = load_vae(; root_folder, dataset_size = training_dataset_size, episode_slicing_interval)
    if "B-PinE" in solver_string_lst
        contingency_game_solver, contingency_game, initial_belief = setup_contingency_game_solver(;
            game, 
            initial_state, 
            ground_truth_parameters = goal_dataset[1],
            horizon,
        )
        number_of_hypotheses = contingency_game_solver.fields.number_of_hypotheses
    end
    # strategy of the ego agent
    receding_horizon_strategy_ego =
        WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = nothing)
    # store results
    result_data = initialize_result_data(; solver_string_lst)

    #==========================#
    # Begin of experiment loop
    #==========================#
    for solver_string in solver_string_lst
        for trial in 1:length(goal_dataset)
            println("#########################\n New Iteration: ", trial, "/", length(goal_dataset), "\n#########################")
            # initialization
            episode_data = initialize_episode_data()
            initial_state = initial_state_set[trial]
            system_state = initial_state
            goal_estimation = nothing
            goal = goal_dataset[trial]

            # strategy of the opponet
            receding_horizon_strategy =
                WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = goal)
            # initial solve for plotting
            strategy = MCPGameSolver.solve_trajectory_game!(receding_horizon_strategy.solver, game, system_state, 
                receding_horizon_strategy; receding_horizon_strategy.solve_kwargs...)
            # strategy.substrategies[ego_agent_id] = dummy_substrategy
            figure = Makie.Figure(resolution = (1200, 900))
            visualization = visualize!(figure, game, system_state, strategy; targets = goal, obstacle_radius = collision_radius,
                ego_agent_id, opponents_id)
            extra_viz = ExtraVisualization(nothing, nothing, nothing)
            if solver_string == "B-PinE"
                initial_belief = prior_belief_from_vae(vae; number_of_hypotheses, system_state, ll, lw, rng, visualization, extra_viz)
                contingency_solution = solve_contingency_game(contingency_game_solver, contingency_game, initial_belief)
                predicted_opponents_strategy = deepcopy(contingency_solution.strategy.substrategies[opponents_id[1]])
                predicted_opponents_trajectory = predicted_opponents_strategy.fields.branch_strategies
                # println("belief left: ", initial_belief[1].weight)
                # println("belief straight: ", initial_belief[2].weight)
            else
                predicted_opponents_strategy = deepcopy(strategy.substrategies[opponents_id])
            end
            display(figure)
            predicted_strategy_visualization = visualize_prediction(predicted_opponents_strategy; visualization, ego_agent_id, rng)

            xs_observation = solver_string == "B-PinE" ? Array{Any}[] : Array{Float64}[]
            xs_pre = BlockArrays.BlockVector{Float64}[] # keep track of initial states for each inverse game solving
            last_solution = nothing # for warm-starting
            erase_last_solution!(receding_horizon_strategy)
            erase_last_solution!(receding_horizon_strategy_ego)

            # Start of the simulation loop
            Makie.record(figure, "data/"*solver_string*"sim_steps"*string(trial)*".mp4", 1:n_sim_steps; framerate = 15) do t
                # opponents' solve
                time_exec_opponents = @elapsed strategy = solve_game_with_resolve!(receding_horizon_strategy, game, system_state)
                #===========================================================#
                # player 2 infers player 1's objective and plans its motion
                if t-1 < vector_size && solver_string != "GT"
                    if solver_string == "B-PinE"
                        time_exec = @elapsed initial_belief = prior_belief_from_vae(vae; number_of_hypotheses, system_state, ll, lw, rng, visualization, extra_viz)
                        contingency_solution = solve_contingency_game_with_warm_start(; contingency_game_solver, contingency_game, initial_belief, last_solution)                        
                        last_solution = contingency_solution.info.raw_solution
                        strategy_ego = contingency_solution.strategy.substrategies[ego_agent_id].fields.branch_strategies[1]
                        apply_ego_contingency_plan!(; strategy, strategy_ego, ego_agent_id, system_state, rng)
                        predicted_opponents_trajectory = deepcopy(contingency_solution.strategy.substrategies[opponents_id[1]].fields.branch_strategies)
                        solving_status = PATHSolver.MCP_Solved
                    end
                else
                    #=================================#                   
                    if solver_string == "B-PinE"
                        time_exec = @elapsed initial_belief = posterior_belief_from_vae(vae; set_up, number_of_hypotheses, system_state, ll, lw, rng, xs_observation, visualization, extra_viz)
                            contingency_solution = solve_contingency_game_with_warm_start(; contingency_game_solver, contingency_game, initial_belief, last_solution)                        
                        contingency_solved = contingency_solution.info.raw_solution.status == PATHSolver.MCP_Solved
                        last_solution = contingency_solved ? contingency_solution.info.raw_solution : nothing
                        if contingency_solved
                    strategy_ego = contingency_solution.strategy.substrategies[ego_agent_id].fields.branch_strategies[1]
                    apply_ego_contingency_plan!(; strategy, strategy_ego, ego_agent_id, system_state, rng)
                        else
                            @info "Contingency game not solved cleanly, emergency braking..."
                            apply_braking!(; strategy, game, system_state, ego_agent_id, horizon, max_acceleration, rng)
                        end
                    predicted_opponents_trajectory = deepcopy(contingency_solution.strategy.substrategies[opponents_id[1]].fields.branch_strategies)
                    #=================================#
                    elseif solver_string == "B-MAP"
                        information_vector = reduce(vcat, xs_observation)
                        reduced_observation = extract_observation(xs_observation; ego_agent_id, num_player, observation_dim = 3)
                        time_exec = @elapsed goal_estimation = sample_goal_from_vae_posterior(vae; set_up, system_state, ll, lw, rng, xs_observation = reduced_observation, visualization)
                        visualize_mle_estimates!(visualization.environment_axis, goal_estimation; t, n_sim_steps)
                        receding_horizon_strategy_ego.context_state = goal_estimation
                        # solve forward game
                        time_forward = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        time_exec += time_forward
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]
                    #=================================#
                    elseif solver_string in ["R-MLE", "BP-MLE"]
                        information_vector = reduce(vcat, xs_observation)
                        #=================================# # liu2023ral (enhanced with vae prior)
                        # very first initialization (later use the previous estimation as warm start)
                        if solver_string == "R-MLE"
                            random_goal = sample_random_initial_guess(; rng, lw, collision_radius, ego_agent_id, goal, num_player)# 
                            initial_estimation = !isnothing(goal_estimation) ? goal_estimation : random_goal
                        elseif solver_string == "BP-MLE"
                            initial_estimation = !isnothing(goal_estimation) ? goal_estimation : sample_goal_from_vae_prior(vae; system_state, num_player, ll, lw, 
                                ego_agent_id, ego_ego = goal[Block(ego_agent_id)], rng, visualization)
                        end
                        # solve inverse game
                        goal_estimation, last_solution, i_, info_, time_exec = interactive_inference_by_backprop(mcp_game, xs_pre[1],
                            information_vector, initial_estimation, goal[Block(ego_agent_id)]; max_grad_steps, lr, 
                            last_solution = last_solution, num_player, ego_agent_id, observation_idx_set, ego_state_idx,
                            lw, collision_radius, visualization,
                        )
                        visualize_mle_estimates!(visualization.environment_axis, goal_estimation; t, n_sim_steps)
                        receding_horizon_strategy_ego.context_state = goal_estimation
                        # solve forward game
                        time_forward = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        time_exec += time_forward
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]
                        #=================================#
                        #=================================#
                    elseif solver_string == "St-BP"
                        if t-1 == vector_size
                            # randomly sampling from the VAE prior as a heuristic estimation of the opoonent's goal
                            time_exec = @elapsed goal_estimation = sample_goal_from_vae_prior(vae; system_state, num_player, ll, lw, 
                                ego_agent_id, ego_ego = goal[Block(ego_agent_id)], rng, visualization)
                        receding_horizon_strategy_ego.context_state = goal_estimation
                        end
                        # solve forward game
                        time_exec = @elapsed strategy_ego = solve_game_with_resolve!(receding_horizon_strategy_ego, game, system_state)
                        println(time_exec, "s")
                        solving_status = check_solver_status!(
                            receding_horizon_strategy_ego, strategy, strategy_ego, game, system_state, 
                            ego_agent_id, horizon, max_acceleration, rng
                        )
                        predicted_opponents_trajectory = strategy_ego.substrategies[opponents_id]     
                    elseif solver_string == "GT"
                        #=================================# # ground truth (game-theoretic interaction in a centralized fashion)
                        time_exec = time_exec_opponents
                        goal_estimation = goal
                        predicted_opponents_trajectory = strategy.substrategies[opponents_id]
                        #=================================#
                    else
                        error("Not a valid solver name!") 
                    end
                end
                #===========================================================#
                # visualize what the ego thinks the opponent will do
                let
                    if t-1 < vector_size && solver_string != "B-PinE"
                        strategy_to_be_visualized = strategy.substrategies[opponents_id]
                    elseif t-1 >= vector_size && solver_string != "B-PinE"
                        strategy_to_be_visualized = predicted_opponents_trajectory
                    else
                        strategy_to_be_visualized = predicted_opponents_trajectory
                    end
                    map(1:length(predicted_strategy_visualization)) do ii
                        if solver_string != "B-PinE"
                            predicted_strategy_visualization[ii][] = strategy_to_be_visualized[ii]
                        else
                            predicted_strategy_visualization[ii][] = 
                                contingency2liftedstrategy(strategy_to_be_visualized[ii], ii, rng)
                        end
                    end
                end

                # update state
                pointmasses_trajectory, control_sequence, _ =
                    rollout(game.dynamics, strategy, system_state, horizon)
                system_state = pointmasses_trajectory[turn_length + 1]
                previous_state = pointmasses_trajectory[turn_length]
                # update observation
                update_observation_vec!(; solver_string, xs_observation, xs_pre, system_state, num_player, ego_agent_id, previous_state, vector_size)
                # visualization
                visualization.strategy[] = strategy
                # visualization.targets[] = goal
                for (x, _) in zip(pointmasses_trajectory, 1)
                    visualization.pointmasses[] = x
                end
                if !isnothing(extra_viz.history)
                    extra_viz.history[] = episode_data.states[(end-vector_size+1):end]
                end
                if t == vector_size + 1 && isnothing(extra_viz.history)
                    extra_viz.history = Makie.Observable(episode_data.states[(end-vector_size+1):end])
                    visualize_history!(visualization.environment_axis, extra_viz.history)
                end
                # store data
                load_episode_data!(; episode_data, solver_string, system_state, control = control_sequence[1].reference_control,
                    goal, time_exec = t>vector_size ? time_exec : nothing, goal_estimation = solver_string == "B-PinE" ? nothing : goal_estimation, 
                    predicted_opponents_trajectory = t>vector_size ? predicted_opponents_trajectory : nothing,
                    belief = solver_string == "B-PinE" ? initial_belief : nothing,
                    solve_status = solver_string == "B-PinE" ? contingency_solution.info.raw_solution.status : nothing)                    
            end
            push!(result_data[solver_string], episode_data)
        end
        if save # save data
            jldsave(root_folder * "monte_carlo_results.jld2"; result_data)
        end
    end
end