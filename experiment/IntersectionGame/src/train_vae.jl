#=============================================================================================

Functions for training a VAE using differentiable game solver in traffic intersection scenario

==============================================================================================#


#=================== generate data by running ground truth interaction simulation =======================#

function run_intersection_data_collection(; 
    number_trials = 1, solver = nothing, num_player = 2, ego_agent_id = 1, 
    lane_id_per_player = [10, 13],
    ll = 2.0, lw = 0.6, turn_radius = 0.3, 
    collision_radius = 0.08, max_velocity = 0.2, max_acceleration = 0.12, max_ϕ = π/4,
    collision_avoidance_coefficient = 400, hard_constraints = false, # collision avoidance inequalities
    rng = Random.MersenneTwister(1), horizon = 15, n_sim_steps = 76,
    vector_size = 15, # number of position points that the ego keeps as observation
    turn_length = 1, # number of steps to take along the MPGP horizon
    max_grad_steps = 10, # max online gradient steps for MLE
    root_folder = "data/",
    visual = false,
    save = false,
)

    #=================================#
    # Construction of game objects
    #=================================#
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)
    vertices, roadway = construct_intersection_roadway(; ll, lw, turn_radius)
    environment = RoadwayEnvironment(vertices, roadway, lane_id_per_player, collision_radius)
    game = construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
        collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ, game_type = "intersection")
    initial_state_set, goal_dataset, target_lane_distribution_opp = construct_intersection_dataset(game, horizon, 
        rng, num_player, ego_agent_id, collision_radius, number_trials; max_velocity, vertices)
    if save
        jldsave(root_folder * "goal_dataset.jld2"; goal_dataset)
    end
    initial_state = initial_state_set[1]
    system_state = initial_state

    state_dimension = state_dim(game.dynamics.subsystems[1])
    control_dimension = control_dim(game.dynamics.subsystems[1])
    ego_state_idx = let
        offset = ego_agent_id != 1 ? sum([blocksizes(initial_state)[1][ii] for ii in 1:(ego_agent_id - 1)]) : 0
        Vector((offset + 1):(offset + blocksizes(initial_state)[1][ego_agent_id]))
    end
    solver_string_lst = ["ground_truth"]
    solver = @something(solver, MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal_dataset[1], 1))) # public solver for the uncontrolled agents
    mcp_game = solver.mcp_game

    observation_opponents_idx_set = construct_observation_index_set(;
        num_player, ego_agent_id, vector_size, state_dimension, mcp_game,
    )
    block_sizes_params = blocksizes(goal_dataset[1]) |> only
    #====================#
    # strategy of the ego agent
    receding_horizon_strategy_ego =
        WarmStartRecedingHorizonStrategy(; solver, game, turn_length, context_state = nothing)
    # a dummy strategy of constant-velocity rollout
    dummy_substrategy, _ = create_dummy_strategy(game, system_state,
        control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng)
    intersection_dataset = []

    #==========================#
    # Begin of experiment loop
    #==========================#
    for solver_string in solver_string_lst
        for trial in 1:length(goal_dataset)
            println("#########################\n New Iteration: ", trial, "/", length(goal_dataset), "\n#########################")
          
            episode_data = []
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
            strategy.substrategies[ego_agent_id] = dummy_substrategy
            if visual
                predicted_opponents_strategy = deepcopy(strategy.substrategies[opponents_id])
                figure = Makie.Figure(resolution = (1200, 900))
                visualization = visualize!(figure, game, system_state, strategy; targets = goal, obstacle_radius = collision_radius,
                    ego_agent_id, opponents_id)
                display(figure)
                predicted_strategy_visualization = visualize_prediction(predicted_opponents_strategy; visualization, ego_agent_id, rng)
            end

            xs_observation = Array{Float64}[]
            xs_pre = BlockArrays.BlockVector{Float64}[] # keep track of initial states for each inverse game solving
            last_solution = nothing # for warm-starting
            erase_last_solution!(receding_horizon_strategy)
            erase_last_solution!(receding_horizon_strategy_ego)

            # Start of the simulation loop
            for t in 1:n_sim_steps
            # Makie.record(figure, "data/"*solver_string*"sim_steps.mp4", 1:n_sim_steps; framerate = 15) do t
                push!(episode_data, system_state)
                # opponents' solve
                time_exec_opponents = @elapsed strategy = solve_game_with_resolve!(receding_horizon_strategy, game, system_state)
                #===========================================================#
                # player 2 infers player 1's objective and plans her motion
                if length(xs_observation) < vector_size && solver_string != "ground_truth"
                    # use a dummy strategy at the first few steps when observation is not sufficient
                    dummy_substrategy, _ = create_dummy_strategy(game, system_state, 
                        control_dim(game.dynamics.subsystems[ego_agent_id]), horizon, ego_agent_id, rng)
                    strategy.substrategies[ego_agent_id] = dummy_substrategy
                    solving_status = PATHSolver.MCP_Solved
                else
                    if solver_string == "backprop"
                        information_vector = reduce(vcat, xs_observation)
                        #=================================# # our solver
                        # very first initialization (later use the previous estimation as warm start)
                        random_goal = mortar([system_state[Block(ii)][2:3] for ii in 1:num_player])
                        random_goal[Block(ego_agent_id)] = goal[Block(ego_agent_id)] # ego goal known

                        initial_estimation = !isnothing(goal_estimation) ? goal_estimation : random_goal
                        # solve inverse game
                        goal_estimation, last_solution, i_, info_, time_exec = interactive_inference_by_backprop(mcp_game, xs_pre[1],
                            information_vector, initial_estimation, goal[Block(ego_agent_id)]; max_grad_steps, lr = 2.1e-2, 
                            last_solution = last_solution, num_player, ego_agent_id, observation_opponents_idx_set, ego_state_idx,
                        )
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
                    elseif solver_string == "ground_truth"
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
                if visual
                    if visualization.skip_button.clicked[]
                        visualization.skip_button.clicked[] = false
                        @info "Manually skipping the episode..."
                        # @goto end_of_episode
                    end
                    while visualization.pause_button.clicked[]
                        sleep(0.1)
                        if visualization.continue_button.clicked[]
                            visualization.pause_button.clicked[] = false
                            visualization.continue_button.clicked[] = false
                        end
                    end
                    if visualization.stop_button.clicked[]
                        visualization.skip_button.clicked[] = false
                        @info "Manually terminating the experiment..."
                        @goto end_of_experiment
                    end

                    # visualize what the ego thinks the opponent will do
                    let
                        if length(xs_observation) < vector_size
                            strategy_to_be_visualized = strategy.substrategies[opponents_id]
                        else
                            strategy_to_be_visualized = predicted_opponents_trajectory
                        end
                        map(1:length(predicted_strategy_visualization)) do ii
                            predicted_strategy_visualization[ii][] = strategy_to_be_visualized[ii]
                        end
                    end
                end

                # update state
                pointmasses_trajectory, control_sequence, _ =
                    rollout(game.dynamics, strategy, system_state, horizon)
                system_state = pointmasses_trajectory[turn_length + 1]
                previous_state = pointmasses_trajectory[turn_length]

                # compute information vector
                push!(xs_observation, reduce(vcat, [system_state[Block(ii)][[1, 2, 4]] for ii in 1:num_player if ii != ego_agent_id])) # partial observation
                # push!(xs_observation, reduce(vcat, [system_state[Block(ii)] for ii in 1:num_player if ii != ego_agent_id])) # full observation
                estimated_state = previous_state # compute_state_estimation(previous_state, system_state, num_player)
                push!(xs_pre, estimated_state)
                if length(xs_observation) > vector_size
                    popfirst!(xs_observation)
                    popfirst!(xs_pre)
                end
                if visual
                    # visualization
                    visualization.strategy[] = strategy
                    # visualization.targets[] = goal
                    for (x, _) in zip(pointmasses_trajectory, 1)
                        visualization.pointmasses[] = x
                    end
                    sleep(0.015)
                end
            end
            push!(intersection_dataset, episode_data)
        end
    end
    if save
        jldsave(root_folder * "intersection_dataset.jld2"; intersection_dataset)
    end
    @label end_of_experiment
end

#=================================== train VAE using generated data =========================#

function train_generative_model_with_driving_data(;
    set_up = construct_training_setup(; dataset_size = 700, episode_slicing_interval = 1)        
)
    # training from scratch
    vae = setup_mcp_vae(set_up)
    # training a pretrained model
    # vae = JLD2.load(set_up.root_folder * "mcp_vae.jld2")["vae"]
    # encoder = JLD2.load(set_up.root_folder * "encoder.jld2")["encoder"]
    # decoder = JLD2.load(set_up.root_folder * "decoder.jld2")["decoder"]
    # vae = setup_mcp_vae(set_up; encoder, decoder)
    Flux.trainmode!(vae)
    train_mcp_vae!(vae; set_up)
    Flux.testmode!(vae)
    # TODO: handle initial guess
end

#======================= training function ===========================#

function train_mcp_vae!(vae::MCP_VAE; set_up)
    training_loss_log = []
    validation_loss_log = []
    for epoch in 1:set_up.training_config.n_epochs
        Flux.testmode!(vae)
        # print current loss
        println("Epoch $epoch")
        loss = get_ELBO_loss(vae; set_up.training_config.device, set_up.rng, set_up.μ_training, set_up.σ_training)
        validation_loss = loss(set_up.validation_data)
        training_loss = loss(set_up.data)
        push!(validation_loss_log, validation_loss)
        push!(training_loss_log, training_loss)
        @info "training_loss: $(training_loss)"
        @info "validation_loss: $(validation_loss)"
        # train model
        Flux.trainmode!(vae)
        @time Flux.train!(loss, params(vae), set_up.data_batch_iterator, set_up.training_config.optimizer)
        Flux.testmode!(vae)
        # save data, plot loss and learned distribution
        if epoch % 50 == 0
            # vae |> cpu
            jldsave(set_up.root_folder * (now() |> string) * "encoder.jld2"; vae.encoder)
            jldsave(set_up.root_folder * (now() |> string) * "decoder.jld2"; vae.decoder)
            jldsave(set_up.root_folder * (now() |> string) * "mcp_vae.jld2"; vae)
            # vae |> set_up.training_config.device
            fig = Makie.Figure()
            axis_val = Makie.Axis(fig[1, 1], title = "Test loss")
            axis_train = Makie.Axis(fig[1, 2], title = "Training loss")
            Makie.lines!(axis_val, Vector{Float32}(1:epoch), Vector{Float32}(validation_loss_log))
            Makie.lines!(axis_train, Vector{Float32}(1:epoch), Vector{Float32}(training_loss_log))
            save(set_up.root_folder * (now() |> string) * "loss_log.png", fig, px_per_unit = 2)
            x_rec = sample_latent_space(vae, set_up)
            gt_goals = [set_up.gt_goals[ii][Block(2)] for ii in 1:length(set_up.data.gt_goals)]
            gt_goals = reduce(hcat, gt_goals)
            fig = plot_comparison_two_distributions(gt_goals, x_rec)
            save(set_up.root_folder * (now() |> string) * "density.png", fig)
        end
    end
end 

function mvnormal_parameters(x)
    in_dim = size(x, 1)
    iseven(in_dim) || throw(ArgumentError("x must have an even first dimension for splitting."))
    out_dim = in_dim ÷ 2
    μ = selectdim(x, 1, 1:out_dim)
    Σ_diag = softplus.(selectdim(x, 1, (out_dim+1):in_dim)) .+ 1e-6
    (; μ, Σ_diag)
end

struct MyNormLayer
    """
    MyNormLayer: scaling the VAE output to a reasonable regime
    """
    lw::Float64
    collision_radius::Float64
end

# normalize output of VAE
function (my_norm_layer::MyNormLayer)(x)
    x = tanh.(x)
    x_length = (my_norm_layer.lw/2)*2 - (-(my_norm_layer.lw/2-my_norm_layer.collision_radius))
    xs = x[1,:] .* (x_length/2)
    xs = xs .+ ((my_norm_layer.lw/2)*2-x_length/2)
    xs = reshape(xs, 1, :)

    y_length = -my_norm_layer.collision_radius - (-(my_norm_layer.lw/2)*2)
    ys = x[2,:] .* (y_length/2)
    ys = ys .+ ((-my_norm_layer.collision_radius)-(y_length/2))
    ys = reshape(ys, 1, :)
    x = vcat(xs, ys)
    x
end

function (model::MCP_VAE)(x, ϵ)
    """
    Forward pass of MCP_VAE
    """
    num_player = num_players(model.mcp_game.game)
    data = x.observations
    
    # pass through the encoder and decoder to produce game objectives
    d = model.encoder(data)
    z = d.μ + d.Σ_diag .* ϵ # latent space samples

    # pass through the decoder + mcp solver to produce game trajectories
    objectives = model.decoder(z)

    solver_statistics = []
    x̂ = Zygote.forwarddiff(objectives; chunk_threshold = length(objectives)) do objectives
        mapreduce(hcat, 1:size(objectives)[2]) do ii
            system_state = x.initial_states[ii]
            inferred_objectives = map(1:num_player) do jj
                jj == 2 ? objectives[:,ii] : x.gt_goals[ii][Block(jj)]
            end |> mortar
            game_solution = MCPGameSolver.solve_mcp_game(model.mcp_game, system_state, inferred_objectives; verbose = false)
            if game_solution.status != PATHSolver.MCP_Solved
                push!(solver_statistics, 0)
            else
                push!(solver_statistics, 1)
            end
            mapreduce(vcat, 1:num_player) do player_id
                if player_id == 1 # TODO: remove hard code
                    sol = game_solution.primals[player_id][model.observation_idx.ego_idx]
                else
                    sol = game_solution.primals[player_id][model.observation_idx.opp_idx]
                end
                sol  
            end
            # reduce(vcat, [game_solution.primals[player_id][model.observation_idx] for player_id in 1:num_player])
        end
    end
    infeasible_percentage = 1- (sum(solver_statistics))/size(objectives)[2]
    println("infeasible_percentage: ", infeasible_percentage)

    (; x̂, z, d)
end

# kl divergence
function kld_from_normal(μ, Σ_diag)
    k = length(μ)
    1 // 2 * (sum(Σ_diag) + sum(μ .* μ) .- k .- sum(log.(prod(Σ_diag; dims=1))))
end

function get_ELBO_loss(model::MCP_VAE; rng, device, μ_training, σ_training)
    function loss(x)
        ϵ = rand(rng, Distributions.Normal(), model.dims.dim_z, size(x.observations, 2)) |> device # reparametrization trick
        x̂, _, d = model(x, ϵ) #|> device
        # undo_norm_obs = x.observations .* (σ_training .+ 1e-5) .+ μ_training # undo observation normalization
        x̂ = (x̂ .- μ_training) ./ (σ_training .+ 1e-5)
        KL_Divergence = kld_from_normal(d.μ, d.Σ_diag)

        (sum((x.observations .- x̂) .^ 2) + KL_Divergence) / size(x.observations, 2)
    end
end

function shuffle_dataset!(set_up)
    # training_data_idx = 1:Int(size(set_up.data.observations)[2]) |> Vector
    Random.shuffle!(set_up.rng, set_up.training_data_idx)
    set_up.data.observations = set_up.data.observations[:, set_up.training_data_idx]
    set_up.data.initial_states = set_up.data.initial_states[set_up.training_data_idx]
    set_up.data.gt_goals = set_up.data.gt_goals[set_up.training_data_idx]
end

mutable struct Dataset
    observations::Matrix{Float64}
    initial_states::Vector{Any}
    gt_goals::Vector{Any}
end

#============================= construct model =========================# 

function setup_mcp_vae(set_up; encoder = nothing, decoder = nothing)
    lw = set_up.mcp_game.game.env.roadway.opts.lane_width #|> set_up.training_config.device
    collision_radius = set_up.mcp_game.game.env.radius #|> set_up.training_config.device
    
    encoder = isnothing(encoder) ? Chain(
        Dense(set_up.dims.dim_x, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, 2 * set_up.dims.dim_z; init = glorot_uniform(set_up.rng)),
        mvnormal_parameters,
    ) : encoder
    decoder = isnothing(decoder) ? Chain(
        Dense(set_up.dims.dim_z, 5 * set_up.dims.dim_z, relu; init = glorot_uniform(set_up.rng)),
        Dense(5 * set_up.dims.dim_z, 5 * set_up.dims.dim_z, relu; init = glorot_uniform(set_up.rng)),
        Dense(5 * set_up.dims.dim_z, set_up.dims.dim_game_objective; init = glorot_uniform(set_up.rng)),
        MyNormLayer(lw, collision_radius),
    ) : decoder
    state_dimension = state_dim(set_up.mcp_game.game.dynamics.subsystems[1])
    opp_idx = mapreduce(vcat, 1:set_up.mcp_game.horizon) do ii
        offset = (ii - 1) * state_dimension
        [offset+1, offset+2, offset+4]
    end
    ego_idx = mapreduce(vcat, 1:set_up.mcp_game.horizon) do ii
        offset = (ii - 1) * state_dimension
        [offset+2]
    end
    observation_idx = (; ego_idx, opp_idx)
    
    MCP_VAE(encoder, decoder, set_up.mcp_game, set_up.dims, observation_idx) |> set_up.training_config.device
end

#============================= construct training dataset from simulated interaction and normalize the dataset ============================#

function construct_training_setup(; root_folder = "data/", dataset_size = 2000,
    num_player = 2,
    ego_agent_id = 1,
    number_trials = 1,
    ll = 2.0, 
    lw = 0.6, 
    turn_radius = 0.3, 
    collision_radius = 0.08, 
    max_velocity = 0.2, 
    max_acceleration = 0.12, 
    max_ϕ = π/4, 
    collision_avoidance_coefficient = 400,
    hard_constraints = false, # collision avoidance inequalities
    rng = Random.MersenneTwister(1), 
    horizon = 15,
    solver = nothing,
    lane_id_per_player = [10, 13],
    observation_dim_per_player = [1, 3],
    shuffle = true,
    episode_slicing_interval = 1,
    )
    training_config = (;
        dataset_size,
        batchsize = 128,
        n_epochs = 1000,
        device = cpu,
        optimizer = Flux.Optimiser(Flux.ClipValue(50), Adam(0.0002))
    )
    trajectories = JLD2.load(root_folder * "intersection_dataset.jld2")["intersection_dataset"][1:dataset_size]
    gt_goals = JLD2.load(root_folder * "goal_dataset.jld2")["goal_dataset"][1:dataset_size]

    # mcp_game = JLD2.load(root_folder * "mcp_game.jld2")["mcp_game"]
    opponents_id = deleteat!(Vector(1:num_player), ego_agent_id)
    vertices, roadway = construct_intersection_roadway(; ll, lw, turn_radius)
    environment = RoadwayEnvironment(vertices, roadway, lane_id_per_player, collision_radius)
    game = construct_game(num_player; min_distance = 2 * collision_radius, hard_constraints, 
        collision_avoidance_coefficient, environment, max_velocity, max_acceleration, max_ϕ, game_type = "intersection")
    initial_state_set, goal_dataset, target_lane_distribution_opp = construct_intersection_dataset(game, horizon, 
        rng, num_player, ego_agent_id, collision_radius, number_trials; max_velocity, vertices)
    solver = @something(solver, MCPCoupledOptimizationSolver(game, horizon, blocksizes(goal_dataset[1], 1)))
    mcp_game = solver.mcp_game

    # construct (partially) observed trajectory slices
    episode_length = trajectories[1] |> length
    trajectory_slices_indices = map(1:((episode_length - 1 - horizon) ÷ episode_slicing_interval + 1)) do ii
        offset = (ii - 1) * episode_slicing_interval
        offset + 2
    end
    initial_states_indices = trajectory_slices_indices .- 1

    dims = (; dim_x = mcp_game.horizon * sum(observation_dim_per_player), dim_z = 16, dim_hidden = 128, dim_game_objective = 2)
    initial_states = mapreduce(vcat, 1:length(trajectories)) do ii
        trajectories[ii][initial_states_indices]
    end #|> training_config.device
    observations = mapreduce(hcat, 1:length(trajectories)) do ii
        mapreduce(hcat, 1:length(trajectory_slices_indices)) do jj
            data = mapreduce(vcat, 1:num_player) do ll
                mapreduce(vcat, 1:horizon) do kk
                    if ll == ego_agent_id
                        obs = [trajectories[ii][trajectory_slices_indices[jj]+kk-1][Block(ll)][2]]
                    else
                        obs = trajectories[ii][trajectory_slices_indices[jj]+kk-1][Block(ll)][[1, 2, 4]]
                    end
                    obs 
                end
            end
        end
    end |> training_config.device
    gt_goals = mapreduce(vcat, 1:length(gt_goals)) do ii
        repeat([gt_goals[ii]], outer = length(trajectory_slices_indices))
    end
    training_data_idx = 1:Int(length(initial_states) * 0.8) |> Vector
    test_data_idx = (Int(length(initial_states) * 0.8) + 1):length(initial_states) |> Vector
    if shuffle
        Random.shuffle!(rng, training_data_idx)
        Random.shuffle!(rng, test_data_idx)
    end
    training_observations = observations[:, training_data_idx] #|> training_config.device
    test_observations = observations[:, test_data_idx] #|> training_config.device
    
    # data normalization
    μ_training = mean(training_observations; dims = 2) #|> training_config.device
    σ_training = std(training_observations; dims = 2) #|> training_config.device
    training_observations = (training_observations .- μ_training) ./ (σ_training .+ 1e-5)
    test_observations = (test_observations .- μ_training) ./ (σ_training .+ 1e-5)

    data = (; observations = training_observations, initial_states = initial_states[training_data_idx], gt_goals = gt_goals[training_data_idx]) # |> training_config.device
    validation_data = (; observations = test_observations, initial_states = initial_states[test_data_idx], gt_goals = gt_goals[test_data_idx]) # |> training_config.device

    data_batch_iterator = Flux.Data.DataLoader(data; training_config.batchsize)

    (; rng, data, validation_data, μ_training, σ_training, dims, training_config, gt_goals, mcp_game, data_batch_iterator, root_folder)
end