#===================================

functions for the MLE baseline

===================================#

function likelihood_cost(τs_observed, goal_estimation, initial_state; mcp_game, observation_idx_set)
    """
    inverse game loss for the MLE baseline
    """
    solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
        goal_estimation; initial_guess = nothing)
    if solution.status != PATHSolver.MCP_Solved
        @info "Inner solve did not converge properly, re-initializing..."
        solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
            goal_estimation; initial_guess = nothing)
    end
    τs_solution = solution.variables[observation_idx_set]
    norm_sqr(τs_observed - τs_solution)
end

function interactive_inference_by_backprop(
    mcp_game, initial_state, τs_observed, 
    initial_estimation, ego_goal; 
    max_grad_steps = 150, lr = 1e-3, last_solution = nothing,
    num_player, ego_agent_id, observation_idx_set,
    ego_state_idx,
    lw = 0.0, collision_radius = 0.0, visualization,
)
    """
    solve inverse game using MLE

    gradient steps using differentiable game solver on the observation likelihood loss
    """
    function likelihood_cost(τs_observed, goal_estimation, initial_state)
        solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
            goal_estimation; initial_guess = last_solution)
        if solution.status != PATHSolver.MCP_Solved
            @info "Inner solve did not converge properly, re-initializing..."
            solution = MCPGameSolver.solve_mcp_game(mcp_game, initial_state, 
                goal_estimation; initial_guess = nothing)
        end
        push!(solving_info, solution.info)
        last_solution = solution.status == PATHSolver.MCP_Solved ? (; primals = ForwardDiff.value.(solution.primals),
        variables = ForwardDiff.value.(solution.variables), status = solution.status) : nothing
        τs_solution = solution.variables[observation_idx_set]
        
        if solution.status == PATHSolver.MCP_Solved
            infeasible_counter = 0
        else
            infeasible_counter += 1
        end
        norm_sqr(τs_observed - τs_solution)
    end
    infeasible_counter = 0
    solving_info = []
    goal_estimation = initial_estimation
    i_ = 0
    time_exec = 0
    for i in 1:max_grad_steps
        i_ = i
        # clip the estimation by the lower and upper bounds
        for ii in 1:num_player
            if ii != ego_agent_id
                goal_estimation[Block(ii)] = clamp.(goal_estimation[Block(ii)], [-(lw/2-collision_radius), -(lw/2)*2], [(lw/2)*2, -collision_radius])
            end
        end
        goal_estimation[Block(ego_agent_id)] = ego_goal
        
        # FORWARD diff
        grad_step_time = @elapsed gradient = Zygote.gradient(τs_observed, goal_estimation, initial_state) do τs_observed, goal_estimation, initial_state
            Zygote.forwarddiff([goal_estimation; initial_state]; chunk_threshold = length(goal_estimation) + length(initial_state)) do θ
                goal_estimation = BlockVector(θ[1:length(goal_estimation)], blocksizes(goal_estimation)[1])
                initial_state = BlockVector(θ[(length(goal_estimation) + 1):end], blocksizes(initial_state)[1])
                likelihood_cost(τs_observed, goal_estimation, initial_state)
            end
        end
        time_exec += grad_step_time
        objective_grad = gradient[2]
        x0_grad = gradient[3]
        x0_grad[ego_state_idx] .= 0 # cannot modify the ego state
        clamp!(objective_grad, -50, 50)
        clamp!(x0_grad, -10, 10)
        objective_update = lr * objective_grad
        x0_update = 1e-3 * x0_grad
        if norm(objective_update) < 1e-4 && norm(x0_update) < 1e-4
            @info "Inner iteration terminates at iteration: "*string(i)
            break
        elseif infeasible_counter >= 4
            @info "Inner iteration reached the maximal infeasible steps"
            break
        end
        goal_estimation -= objective_update
        initial_state -= x0_update
    end
    (; goal_estimation, last_solution, i_, solving_info, time_exec)
end

function plot_mle_heatmap(; t, τs_observed, initial_estimation, initial_state, lw, ll, system_state,
    collision_radius, fig, mcp_game, observation_idx_set, root_folder, mle_estimate)
    """
    Plot cost landscape of the MLE baseline
    """
    @info "start plotting MLE cost landscape..."

    axis = t == 27 ? Makie.Axis(fig[1,1]) : Makie.Axis(fig[1,2])
    # [-(lw/2-collision_radius), -(lw/2)*2], [(lw/2)*2, -collision_radius]
    
    # x = -(lw/2-collision_radius):0.005:(lw/2)*2 |> Vector
    # y =  -(lw/2)*2:0.005:-collision_radius |> Vector
    y_lb = -0.95
    y_ub = 0.95
    x_lb = -0.6
    x_ub = 0.6
    x = x_lb:0.02:x_ub |> Vector
    y =  y_lb:0.02:y_ub |> Vector

    goal_estimates = []
    xs = Vector{Float64}()
    ys = Vector{Float64}()
    for ii in 1:length(x)
        for jj in 1:length(y)
            initial_estimation[Block(2)][1] = x[ii]
            initial_estimation[Block(2)][2] = y[jj]
            push!(goal_estimates, deepcopy(initial_estimation))
            push!(xs, x[ii])
            push!(ys, y[jj])
        end
    end
    cost_values = Vector{Float64}()
    for goal_estimation in goal_estimates
        cost_value = likelihood_cost(τs_observed, goal_estimation, initial_state; mcp_game, observation_idx_set)
        push!(cost_values, cost_value)
    end
    hm = Makie.heatmap!(axis, xs, ys, cost_values; colorscale = log10, interpolate = true, overdraw = true)

    for player_i in 1:blocksize(system_state, 1)
        player_color = player_i == 1 ? parse(Colorant, "rgba(238, 29, 37, 1.0)") : parse(Colorant, "rgba(46,139,87, 1.0)")
        position = system_state[Block(player_i)][1:2]
        rotation = system_state[Block(player_i)][4]
        println(position)
        println(rotation)
        Makie.scatter!(axis, position[1], position[2]; rotation, color = player_color, marker = load("media/car"*string(player_i)*".png"), markersize = 75)
    end
    x1 = [ x_lb, lw/2]
	x2 = [ x_ub, lw/2]
	x3 = [ x_lb, -lw/2]
	x4 = [ x_ub, -lw/2]
	x5 = [ -lw/2, y_ub]
	x6 = [ lw/2, y_ub]
	x7 = [ -lw/2, y_lb]
	x8 = [ lw/2, y_lb]
	x9 = [ -lw/2, lw/2]
	x10 = [ lw/2, lw/2]
	x11 = [ -lw/2, -lw/2]
	x12 = [ lw/2, -lw/2]
	x13 = [ x_lb, 0.]
	x14 = [ -lw/2, 0.]
	x15 = [ lw/2, 0.]
	x16 = [ x_ub, 0.]
	x17 = [ 0., y_ub]
	x18 = [ 0., lw/2]
	x19 = [ 0., -lw/2]
	x20 = [ 0., y_lb]

    Makie.lines!(axis, [x5[1], x9[1]], [x5[2], x9[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x11[1], x7[1]], [x11[2], x7[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x1[1], x9[1] .+ 0.0045], [x1[2], x9[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x3[1], x11[1] .+ 0.0045], [x3[2], x11[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x6[1], x10[1]], [x6[2], x10[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x12[1], x8[1]], [x12[2], x8[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x10[1] .- 0.0045, x2[1]], [x10[2], x2[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x12[1] .- 0.0045, x4[1]], [x12[2], x4[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 10)
    Makie.lines!(axis, [x17[1], x18[1]], [x17[2], x18[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(axis, [x13[1], x14[1]], [x13[2], x14[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(axis, [x15[1], x16[1]], [x15[2], x16[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)
    Makie.lines!(axis, [x19[1] .- 0.0045, x20[1]], [x19[2], x20[2]]; color = colorant"rgba(39, 126, 192, 1.0)", linewidth = 4, linestyle = :dash)

    for ii in 1:Int(length(τs_observed)/3)
        offset = (ii-1)*3
        Makie.scatter!(axis, τs_observed[(offset+1)], τs_observed[(offset+2)]; color = colorant"orange", marker = '*', markersize = 50)
    end
    n_sim_steps = 76
    markersize_range = [4, 25]
    Δ = (markersize_range[2]-markersize_range[1])/(n_sim_steps)
    markersize = markersize_range[1] + Δ*t
    Makie.scatter!(axis, mle_estimate[Block(2)][1], mle_estimate[Block(2)][2]; marker = :star4, markersize = 3*markersize, color = colorant"green")
    if t == 46
        Makie.Colorbar(fig[1, 3], hm, label = "-log p(y | θ)"; size = 50)
        # Makie.hidedecorations!(fig[1,1])
        # Makie.hidedecorations!(fig[1,2])
    end
    Makie.hidedecorations!(axis)
    Makie.save(root_folder*"mle_cost_"*string(t)*".png", fig)
end