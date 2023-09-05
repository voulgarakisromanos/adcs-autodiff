using SatelliteToolboxGeomagneticField
using SatelliteDynamics
using Flux, Zygote
using Enzyme

include("dynamics.jl")
include("utils.jl")
include("parameters.jl")
include("mekf.jl")
include("noise_models.jl")
include("simulation.jl")
include("plots.jl")


inertia_matrix = diagm([1.0;1.5;0.5])
x0 = [1.0;0;0;0;0;0;0]
P = 1.0 * Matrix{Float64}(I, 6, 6)
Q = 0.1 * Matrix{Float64}(I, 6, 6)
R = 0.1 * Matrix{Float64}(I, 6, 6)
sigma_u = 1e-6
sigma_v = 1e-6
mag_noise = 1e-6
sun_noise = 1e-6
dt = 0.1
params = parameter_struct(inertia_matrix, sigma_u, sigma_v, mag_noise, sun_noise, dt, x0, P, Q, R)

# (groundtruth_state_history, state_estimation_history) = run_full_simulation(params)
# plot_histories(groundtruth_state_history, state_estimation_history) 

(q_history, w_history, bias_history, mag_eci, sun_eci, mag_noisy_history, sun_noisy_history, gyro_noisy_history) = run_groundtruth_simulation(params)
gt_target = [q_history;bias_history]

function objective_function(x)
    Q = reshape(x[1:36],6,6)
    R = reshape(x[37:72],6,6)
    diff_params = (Q,R)
    state_estimation_history = run_filter_simulation(diff_params, params, mag_noisy_history, sun_noisy_history, gyro_noisy_history)
    loss = log(Flux.mse(gt_target, state_estimation_history))
end

# Initial guess for x
x = rand(72)  # `param` makes x a `TrackedArray` that Flux can work with.

# Optimizer
opt = ADAM(0.1)  # Gradient descent with learning rate 0.1

# Number of iterations
n_iterations = 10

# Optimization loop
for i in 1:n_iterations
    # Compute the gradient
    grad = Zygote.gradient(objective_function, x)[1]

    # Update x with the optimizer
    Flux.Optimise.update!(opt, x, grad)

    # Calculate current loss
    loss = objective_function(x)

    # Optionally, print the current value of x and the loss
    println("Iteration $i: x = $x, loss = $loss")
end


