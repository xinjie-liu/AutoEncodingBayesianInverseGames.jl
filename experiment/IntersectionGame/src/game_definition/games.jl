#==========================================

Definition of a few traffic games

==========================================#

function intersection_game(
    num_players;
    environment,
    min_distance = 1.0,
    hard_constraints = true,
    collision_avoidance_coefficient = 0,
    dynamics = planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    cost = let
        function target_cost(x, context_state)
            # px, py, v, θ = x
            my_norm_sqr(x[1:2] - context_state[1:2])
        end
        function control_cost(u)
            a, δ = u
            my_norm_sqr(u)
        end
        function collision_cost(x, i)
            cost = map([1:(i - 1); (i + 1):num_players]) do paired_player
                max(0.0, min_distance + 0.0225 - my_norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^3
            end
            total_cost = sum(cost) 
        end
        TimeSeparableTrajectoryGameCost(
            mean,
            GeneralSumCostStructure(),
            1.0,
        ) do x, u, t, θ
            # TODO: continue here: convert cost function to be time separable,
            # We need this so that we can disable beyond a specific stage in the contingency game solver
            map(1:num_players) do i
                if :cost_parameters in fieldnames(typeof(θ))
                    cost = 1.0 * target_cost(x[Block(i)], θ.cost_parameters[Block(i)]) +
                                0.1 * control_cost(u[Block(i)]) +
                                collision_avoidance_coefficient * collision_cost(x, i)
                else
                    cost = 1.0 * target_cost(x[Block(i)], θ[Block(i)]) +
                                0.1 * control_cost(u[Block(i)]) +
                                collision_avoidance_coefficient * collision_cost(x, i) 
                end
            end
        end
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    coupling_constraints = hard_constraints ? 
        shared_collision_avoidance_coupling_constraints(num_players, min_distance) : nothing
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function highway_game(
    num_players;
    environment,
    min_distance=1.0,
    hard_constraints=true,
    collision_avoidance_coefficient=0,
    dynamics=planar_double_integrator(;
        state_bounds=(; lb=[-Inf, -Inf, -0.8, -0.8], ub=[Inf, Inf, 0.8, 0.8]),
        control_bounds=(; lb=[-10, -10], ub=[10, 10]),
    ),
)
    ll = environment.roadway.opts.lane_length
    cost = let
        function target_cost(x, context_state)
            px, py, v, θ = x
            1.0 * (py - context_state[1])^2 + ((1.0 - tanh(4 * (px - (ll - 1.65)))) / 2) * (v * cos(θ) - context_state[2])^2 + 0.2 * θ^2
        end
        function control_cost(u)
            a, δ = u
            a^2 + δ^2
        end
        function collision_cost(x, i)
            cost = map([1:(i-1); (i+1):num_players]) do paired_player
                max(0.0, min_distance + 0.0225 - my_norm(x[Block(i)][1:2] - x[Block(paired_player)][1:2]))^3 # with coefficient of 500
            end
            total_cost = sum(cost)
        end

        TimeSeparableTrajectoryGameCost(
            mean,
            GeneralSumCostStructure(),
            1.0,
        ) do x, u, t, θ
            # TODO: continue here: convert cost function to be time separable,
            # We need this so that we can disable beyond a specific stage in the contingency game solver
            map(1:num_players) do i
                collision_avoidance_coefficient = i == 2 ? 0 : collision_avoidance_coefficient
                1.0 * target_cost(x[Block(i)], θ.cost_parameters[Block(i)]) +
                0.1 * control_cost(u[Block(i)]) +
                collision_avoidance_coefficient * collision_cost(x, i)
            end
        end
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    coupling_constraints = hard_constraints ?
                           shared_collision_avoidance_coupling_constraints(num_players, min_distance) : nothing
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

function one_dim_highway_game(
    num_players;
    environment,
    min_distance = 1.0,
    hard_constraints = true,
    collision_avoidance_coefficient = 0,
    dynamics = one_dim_planar_double_integrator(;
        state_bounds = (; lb = [-Inf, -Inf, -0.8, -0.8], ub = [Inf, Inf, 0.8, 0.8]),
        control_bounds = (; lb = [-10, -10], ub = [10, 10]),
    ),
)
    cost = let
        function target_cost(x, context_state)
            px, py, vx, vy = x
            my_norm_sqr(vx - context_state[1])
        end
        function control_cost(u)
            ax, ay = u
            my_norm_sqr(ax)
        end
        function collision_cost(x, i)
            cost = map([1:(i - 1); (i + 1):num_players]) do paired_player
                max(0.0, min_distance + 0.1 * min_distance - my_norm(x[Block(i)][1] - x[Block(paired_player)][1]))^3
            end
            total_cost = sum(cost) 
        end
        function cost_for_player(i, xs, us, context_state)
            mean_target = mean(map(xs) do x
                target_cost(x[Block(i)], context_state[Block(i)])
            end)
            control = mean(map(us) do u
                control_cost(u[Block(i)])
            end)
            safe_distance_violation = mean(map(xs) do x
                collision_cost(x, i)
            end)
            # front player does not have collision avoidance responsibility
            collision_avoidance_coefficient = i == 2 ? 0 : collision_avoidance_coefficient
            1.0 * mean_target + 0.1 * control + collision_avoidance_coefficient * safe_distance_violation
        end
        function cost_function(xs, us, context_state)
            num_players = blocksize(xs[1], 1)
            [cost_for_player(i, xs, us, context_state) for i in 1:num_players]
        end
        TrajectoryGameCost(cost_function, GeneralSumCostStructure())
    end
    dynamics = ProductDynamics([dynamics for _ in 1:num_players])
    coupling_constraints = hard_constraints ? 
        shared_collision_avoidance_coupling_constraints(num_players, min_distance) : nothing
    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

#====================== dynamics =============================#

# An 1-dim planar double integrator where only state along x-axis is controllable
function one_dim_planar_double_integrator(; dt = 0.1, m = 1, kwargs...)
    dt2 = 0.5 * dt * dt
    # Layout is x := (px, py, vx, vy) and u := (Fx, Fy).
    time_invariant_linear_dynamics(;
        A = [
            1.0 0.0 dt 0.0
            0.0 1.0 0.0 0.0
            0.0 0.0 1.0 0.0
            0.0 0.0 0.0 1.0
        ],
        B = [
            dt2 0.0
            0.0 0.0
            dt 0.0
            0.0 0.0
        ] / m,
        kwargs...,
    )
end
