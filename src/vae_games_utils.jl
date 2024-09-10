#=====================================================

Some utility functions for VAE-differentialbe game

=====================================================#

struct MCP_VAE
    encoder::Any
    decoder::Any
    mcp_game::Any
    dims::Any
    observation_idx::Any
end

@functor MCP_VAE (encoder, decoder)

function load_vae(; root_folder, dataset_size, episode_slicing_interval)
    encoder = JLD2.load(root_folder * "encoder.jld2")["encoder"]
    decoder = JLD2.load(root_folder * "decoder.jld2")["decoder"]
    set_up = construct_training_setup(; dataset_size, episode_slicing_interval)
    vae = setup_mcp_vae(set_up; encoder, decoder)
    Flux.testmode!(vae)
    (; vae, set_up)
end

function visualize_vae_inference!(axis, extra_viz; vae_samples, sample_mean1, sample_mean2, weight1, weight2)
    if isnothing(extra_viz.VAE_samples)
        extra_viz.VAE_samples = Makie.Observable(vae_samples)
        num_samples = (extra_viz.VAE_samples[] |> size)[2]
        for ii in 1:num_samples
            sample = Makie.@lift Makie.Point2f($(extra_viz.VAE_samples)[:,ii])
            Makie.scatter!(axis, sample; color = (:blue, 0.02), markersize = 10)
        end
    else
        extra_viz.VAE_samples[] = vae_samples
    end

    weights = [weight1, weight2] # TODO: real-time transparency
    if isnothing(extra_viz.goal_estimates)
        extra_viz.goal_estimates = Makie.Observable([sample_mean1 |> vec, sample_mean2 |> vec])
        for ii in 1:2
            sample_mean = Makie.@lift Makie.Point2f($(extra_viz.goal_estimates)[ii])
            Makie.scatter!(axis, sample_mean; color = (:green), markersize = 20, marker = :star4)
        end
    else
        extra_viz.goal_estimates[] = [sample_mean1 |> vec, sample_mean2 |> vec]
    end
end

# extract a prior distribution from VAE samples
function prior_belief_from_vae(vae; number_of_hypotheses, system_state, ll, lw, rng, visualization, plot=true, extra_viz = nothing)
    # version 1: heuristic belief extraction
    prior_latent_samples = sample_latent_space(vae; rng, num_samples = 1000)
    num_samples_left_turn = sum(prior_latent_samples[1, :] .>= lw/2.5)
    mean_samples_left_turn = mean(prior_latent_samples[:, prior_latent_samples[1, :] .>= lw/2]; dims = 2)
    num_samples_straight = sum(prior_latent_samples[1, :] .<= 0)
    mean_samples_straight = mean(prior_latent_samples[:, prior_latent_samples[1, :] .<= 0]; dims = 2)
    weight1 = num_samples_left_turn / (num_samples_left_turn + num_samples_straight)
    weight2 = num_samples_straight / (num_samples_left_turn + num_samples_straight)
    if plot
        visualize_vae_inference!(visualization.environment_axis, extra_viz; vae_samples = prior_latent_samples, 
            sample_mean1 = mean_samples_left_turn, sample_mean2 = mean_samples_straight, weight1, weight2)
    end
    belief = [
        (;
            weight= ii == 1 ? weight1 : weight2,
            state=system_state,
            # cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [(3/2)*(lw/2), -lw/4]]) : mortar([[lw/4, ll/4], [-lw/4, -(3/2)*(lw/2)]]),
            cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [mean_samples_left_turn[1], mean_samples_left_turn[2]]]) : mortar([[lw/4, ll/4], [mean_samples_straight[1], mean_samples_straight[2]]]),
            dynamics_parameters=nothing,
        ) for ii in 1:number_of_hypotheses
    ]


    # # version 2: K-means belief extraction
    # θ = sample_latent_space(vae; rng, num_samples = 1000)
    # C = kmeans(θ, 2;)
    # weight1 = round(C.counts[1] / sum(C.counts), digits=4)
    # weight2 = round(C.counts[2] / sum(C.counts), digits=4)        
    # Makie.scatter!(visualization.environment_axis, C.centers[1, 1], C.centers[2, 1])
    # Makie.scatter!(visualization.environment_axis, C.centers[1, 2], C.centers[2, 2])
    
    # println("kmeans 1: ", C.centers[:,1], " belief: ", weight1)
    # println("kmeans 2: ", C.centers[:, 2], " belief: ", weight2)

    # belief = [
    #     (;
    #         weight= ii == 1 ? weight1 : weight2,
    #         state=system_state,
    #         # cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [(3/2)*(lw/2), -lw/4]]) : mortar([[lw/4, ll/4], [-lw/4, -(3/2)*(lw/2)]]),
    #         cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], C.centers[:, 1]]) : mortar([[lw/4, ll/4], C.centers[:, 2]]),
    #         dynamics_parameters=nothing,
    #     ) for ii in 1:number_of_hypotheses
    # ]   
end
             
# extract a posterior distribution from VAE samples
function posterior_belief_from_vae(vae; set_up, number_of_hypotheses, system_state, ll, lw, rng, xs_observation, visualization, extra_viz = nothing, plot=true)
    
    information_vector = mapreduce(vcat, 1:length(xs_observation)) do ii
        reduce(vcat, xs_observation[ii])
    end
    μ_training = set_up.μ_training
    σ_training = set_up.σ_training
    # normalization
    information_vector = (information_vector .- μ_training) ./ (σ_training .+ 1e-5)
    θ, d, z = sample_from_posterior_latent_space(vae, information_vector; rng, num_samples = 1000)
    
    # version 1: using heuristic to get belief
    # num_samples_left_turn = sum(θ[1, :] .>= lw/2)
    # mean_samples_left_turn = mean(θ[:, θ[1, :] .>= lw/2]; dims = 2)
    # num_samples_straight = sum(θ[1, :] .<= 0)
    # mean_samples_straight = mean(θ[:, θ[1, :] .<= 0]; dims = 2)
    # weight1 = num_samples_left_turn / (num_samples_left_turn + num_samples_straight)
    # weight2 = num_samples_straight / (num_samples_left_turn + num_samples_straight)
    # Makie.scatter!(visualization.environment_axis, mean_samples_left_turn[1], mean_samples_left_turn[2])
    # Makie.scatter!(visualization.environment_axis, mean_samples_straight[1], mean_samples_straight[2])
    # println("heuristic left: ", mean_samples_left_turn, " belief: ", weight1)
    # println("heuristic straight: ", mean_samples_straight, " belief: ", weight2) 
    # belief = [
    #     (;
    #         weight= ii == 1 ? weight1 : weight2,
    #         state=system_state,
    #         # cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [(3/2)*(lw/2), -lw/4]]) : mortar([[lw/4, ll/4], [-lw/4, -(3/2)*(lw/2)]]),
    #         cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [mean_samples_left_turn[1], mean_samples_left_turn[2]]]) : mortar([[lw/4, ll/4], [mean_samples_straight[1], mean_samples_straight[2]]]),
    #         dynamics_parameters=nothing,
    #     ) for ii in 1:number_of_hypotheses
    # ]


    # version 2: using K-means to get belief
    C = kmeans(θ, 2;)
    weight1 = round(C.counts[1] / sum(C.counts), digits=4)
    weight2 = round(C.counts[2] / sum(C.counts), digits=4)
    if plot      
        visualize_vae_inference!(visualization.environment_axis, extra_viz; vae_samples = θ, 
            sample_mean1 = C.centers[:,1], sample_mean2 = C.centers[:,2], weight1, weight2)
        # Makie.scatter!(visualization.environment_axis, C.centers[1, 1], C.centers[2, 1])
        # Makie.scatter!(visualization.environment_axis, C.centers[1, 2], C.centers[2, 2])
    end
    println("kmeans 1: ", C.centers[:,1], " belief: ", weight1)
    println("kmeans 2: ", C.centers[:, 2], " belief: ", weight2)

    belief = [
        (;
            weight= ii == 1 ? weight1 : weight2,
            state=system_state,
            # cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], [(3/2)*(lw/2), -lw/4]]) : mortar([[lw/4, ll/4], [-lw/4, -(3/2)*(lw/2)]]),
            cost_parameters=ii == 1 ? mortar([[lw/4, ll/4], C.centers[:, 1]]) : mortar([[lw/4, ll/4], C.centers[:, 2]]),
            dynamics_parameters=nothing,
        ) for ii in 1:number_of_hypotheses
    ]    
end

function sample_goal_from_vae_prior(vae; system_state, num_player, number_of_hypotheses = 2, ll, lw, ego_agent_id, 
        ego_ego = goal[Block(ego_agent_id)], rng, visualization)
    prior_belief = prior_belief_from_vae(vae; number_of_hypotheses, system_state, ll, lw, rng, visualization, plot=false)
    cost_parameters = [prior_belief[1].cost_parameters, prior_belief[2].cost_parameters]
    weights = [prior_belief[1].weight, prior_belief[2].weight]
    goal_estimation = cost_parameters[rand(rng, Distributions.Categorical(weights))]
    Makie.scatter!(visualization.environment_axis, goal_estimation[Block(2)][1], goal_estimation[Block(2)][2])
    goal_estimation
end

function sample_goal_from_vae_posterior(vae; set_up, number_of_hypotheses = 2, system_state, ll, lw, rng, xs_observation, visualization)
    posterior_belief = posterior_belief_from_vae(vae; set_up, number_of_hypotheses, system_state, 
        ll, lw, rng, xs_observation, visualization, plot = false)
    cost_parameters = [posterior_belief[1].cost_parameters, posterior_belief[2].cost_parameters]
    weights = [posterior_belief[1].weight, posterior_belief[2].weight]
    max_weight, max_weight_idx = findmax(weights)
    goal_estimation = cost_parameters[max_weight_idx]
    Makie.scatter!(visualization.environment_axis, goal_estimation[Block(2)][1], goal_estimation[Block(2)][2])
    goal_estimation
end

function plot_comparison_two_distributions(dataset1, dataset2)
    fig = Makie.Figure(resolution = (2000, 800), fontsize = 35)
    colors = [colorant"rgba(105, 105, 105, 0.008)", colorant"rgba(254, 38, 37, 0.008)"]
    ax1 = Axis(fig[1, 1], title=" ground truth vs. VAE-inferred objective distributions", 
        xlabel = "objective: x", ylabel = "objective: y", spinewidth=3)
    Makie.scatter!(ax1, dataset1[1, :], dataset1[2, :], color = colors[1], strokewidth = 5, 
        strokecolor = colorant"rgba(105, 105, 105, 1.0)", label = "ground truth")
    Makie.scatter!(ax1, dataset2[1, :], dataset2[2, :], color = colors[2], strokewidth = 5, 
        strokecolor = colorant"rgba(254, 38, 37, 1.0)", label = "ours")
    Makie.axislegend(ax1, position = :lt)

    ax2 = Axis(fig[1, 2], title="VAE-inferred objective heatmap",
        xlabel = "objective: x", ylabel = "objective: y", spinewidth=3)
    counts = fit(Histogram, (dataset2[1,:] |> vec, dataset2[2,:] |> vec), nbins = 10)
    heatmap!(ax2, counts, colorrange = (0, maximum(counts.weights)))
    # Makie.ylims!(ax1, 0, 0.12)
    fig
end

function sample_from_posterior_latent_space(model::MCP_VAE, observation; set_up = nothing, rng = nothing, num_samples = 20_000)
    """
    Given observed trajectories, sample extensively from the posterior latent distribution p(z | x) and map the samples to game objectives
    """
    if !isnothing(set_up)
        rng = set_up.rng
    end
    d = model.encoder(observation)
    ϵ = rand(rng, Distributions.Normal(), model.dims.dim_z, num_samples) # noise samples
    z = d.μ .+ d.Σ_diag .* ϵ # latent space samples
    θ = model.decoder(z) # inferred objectives
    (; θ, d, z)
end

function reset_solver!(receding_horizon_strategy, goal)
    receding_horizon_strategy.last_solution = nothing
    receding_horizon_strategy.solution_status = nothing
    receding_horizon_strategy.context_state = goal
end

function sample_latent_space(vae, setup = nothing; rng = nothing, num_samples = 10000)
    ```
    Sample extensively from the latent space and then decode
    ```
    if !isnothing(setup)
        rng = setup.rng
    end
    z = rand(rng, Distributions.MultivariateNormal(zeros(vae.dims.dim_z), LinearAlgebra.I(vae.dims.dim_z)), num_samples) #|> setup.training_config.device
    z = reshape(z, vae.dims.dim_z, :)
    μx_z = vae.decoder(z)
end

#============================ debug & evaluation ==============================#

function undo_data_normalization(data, normalization_params)
    μ = normalization_params.μ
    σ = normalization_params.σ
    data .* (σ .+ 1e-5) .+ μ
end