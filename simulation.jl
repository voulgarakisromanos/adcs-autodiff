include("dynamics.jl")
include("utils.jl")
include("parameters.jl")
include("mekf.jl")
include("noise_models.jl")

function run_full_simulation(params)
    torque = zeros(3,1)
    w = [0.01,-0.01,0.02]
    q = [1.0,0,0,0]
    # Define Julian Date (for example purposes, use 2459921.0)
    JD = 2459921.0
    # Declare simulation initial Epoch
    epc0 = Epoch(jd_to_caldate(JD)...)
    # Declare initial state in terms of osculating orbital elements
    oe0  = [R_EARTH + 500e3, 0.01, 75.0, 45.0, 30.0, 0.0]
    
    # Convert osculating elements to Cartesean state
    eci0 = sOSCtoCART(oe0, use_degrees=true)
    
    # Set the propagation end time to one orbit period after the start
    T    = orbit_period(oe0[1])
    epcf = epc0 + T
    
    # Initialize State Vector
    orb  = EarthInertialState(epc0, eci0, dt=1.0,
            mass=100.0, n_grav=20, m_grav=20,
            drag=true, srp=true,
            moon=true, sun=true,
            relativity=false
    )
    
    # Propagate the orbit
    t, epc, eci = sim!(orb, epcf)
    r_eci = eci[1:3,:]
    r_eci = [r_eci[:,i] for i in 1:size(r_eci,2)]
    # v_eci = eci[4:6,:]
    
    rotation_eci2ecef = rECItoECEF.(epc)
    r_ecef = [rotation_eci2ecef[i] * r_eci[i] for i in 1:size(rotation_eci2ecef,1)]
    rotation_ecef2eci = rECEFtoECI.(epc)
    
    mag_ecef = geomagnetic_dipole_field.(r_ecef)
    mag_ecef = [mag_ecef[i] ./ norm(mag_ecef[i]) for i in 1:size(mag_ecef,1)]
    
    sun_eci = sun_position.(epc)
    sun_eci = [sun_eci[i] ./ norm(sun_eci[i]) for i in 1:size(sun_eci,1)]
    
    w_history = Array{Float64}(undef, 3, length(epc))
    q_history = Array{Float64}(undef, 4, length(epc))
    bias_history = Array{Float64}(undef, 3, length(epc))
    state_estimation_history =  Array{Float64}(undef, 7, length(epc))
    
    kf = KalmanFilter(
        params.x0,             # x (initial state)
        params.P,             # P (initial state covariance)
        transition_function,             # F (state transition model)
        transition_function_jacobian,
        params.Q,        # Q (process noise covariance)
        measurement_function, # H (observation model)
        measurement_fun_jacobian,
        params.R,        # R (observation noise covariance)
        params.dt             # I (identity matrix)
    )
    # Dynamics
    
    for i in 1:length(epc)
        w,q = rk4(params.inertia_matrix, w, torque, q, params.dt)
        w_history[:,i] = w
        q_history[:,i] = q
    end
    
    q_history_vectors = [q_history[:,i] for i in 1:size(q_history,2)]
    mag_eci = rotate_vector_by_quaternion.(mag_ecef, q_history_vectors)
    
    # Estimation
    bias = 0.0
    
    for i in 1:length(epc)
        mag_noisy, sun_noisy, gyroscope_measurement, bias = get_noisy_measurements(q_history[:,i], w_history[:,i], bias, mag_eci[i], sun_eci[i], params)
        bias_history[:,i] = bias
        update!(kf, (mag_noisy,sun_noisy), (mag_eci[i],sun_eci[i]))
        predict!(kf, gyroscope_measurement)
        state_estimation_history[:,i] = kf.global_state
    end

    groundtruth_state_history = [q_history; bias_history] # quaternion - bias
    return (groundtruth_state_history, state_estimation_history)
end

function run_groundtruth_simulation(params)
    torque = zeros(3,1)
    w = [0.001,-0.001,0.002]
    q = [1.0,0,0,0]
    # Define Julian Date (for example purposes, use 2459921.0)
    JD = 2459921.0
    # Declare simulation initial Epoch
    epc0 = Epoch(jd_to_caldate(JD)...)
    # Declare initial state in terms of osculating orbital elements
    oe0  = [R_EARTH + 500e3, 0.01, 75.0, 45.0, 30.0, 0.0]
    
    # Convert osculating elements to Cartesean state
    eci0 = sOSCtoCART(oe0, use_degrees=true)
    
    # Set the propagation end time to one orbit period after the start
    T    = orbit_period(oe0[1])
    epcf = epc0 + T./1000
    
    # Initialize State Vector
    orb  = EarthInertialState(epc0, eci0, dt=1.0,
            mass=100.0, n_grav=20, m_grav=20,
            drag=true, srp=true,
            moon=true, sun=true,
            relativity=false
    )
    
    # Propagate the orbit
    t, epc, eci = sim!(orb, epcf)
    r_eci = eci[1:3,:]
    r_eci = [r_eci[:,i] for i in 1:size(r_eci,2)]
    # v_eci = eci[4:6,:]
    
    rotation_eci2ecef = rECItoECEF.(epc)
    r_ecef = [rotation_eci2ecef[i] * r_eci[i] for i in 1:size(rotation_eci2ecef,1)]
    rotation_ecef2eci = rECEFtoECI.(epc)
    
    mag_ecef = geomagnetic_dipole_field.(r_ecef)
    mag_ecef = [mag_ecef[i] ./ norm(mag_ecef[i]) for i in 1:size(mag_ecef,1)]
    
    sun_eci = sun_position.(epc)
    sun_eci = [sun_eci[i] ./ norm(sun_eci[i]) for i in 1:size(sun_eci,1)]
    
    w_history = Array{Float64}(undef, 3, length(epc))
    q_history = Array{Float64}(undef, 4, length(epc))
    bias_history = Array{Float64}(undef, 3, length(epc))
    state_estimation_history =  Array{Float64}(undef, 7, length(epc))
    
    # Dynamics
    
    for i in 1:length(epc)
        w,q = rk4(params.inertia_matrix, w, torque, q, params.dt)
        w_history[:,i] = w
        q_history[:,i] = q
    end
    
    q_history_vectors = [q_history[:,i] for i in 1:size(q_history,2)]
    mag_eci = rotate_vector_by_quaternion.(mag_ecef, q_history_vectors)
    
    # Estimation
    bias = 0.0
    mag_noisy_history  = Array{Float64}(undef, 3, length(epc))
    sun_noisy_history  = Array{Float64}(undef, 3, length(epc))
    gyro_noisy_history  = Array{Float64}(undef, 3, length(epc))


    for i in 1:length(epc)
        mag_noisy_history[:,i], sun_noisy_history[:,i], gyro_noisy_history[:,i], bias = get_noisy_measurements(q_history[:,i], w_history[:,i], bias, mag_eci[i], sun_eci[i], params)
        bias_history[:,i] = bias
    end
    groundtruth_state_history = (q_history, w_history, bias_history, mag_eci, sun_eci, mag_noisy_history, sun_noisy_history, gyro_noisy_history)
    return groundtruth_state_history
end

function run_filter_simulation(diff_params, params, mag_noisy, sun_noisy, gyroscope_measurement)

    kf = KalmanFilter(
        # params.x0,             # x (initial state)
        # params.P,             # P (initial state covariance)
        transition_function,             # F (state transition model)
        transition_function_jacobian,
        diff_params[1],        # Q (process noise covariance)
        measurement_function, # H (observation model)
        measurement_fun_jacobian,
        diff_params[2],        # R (observation noise covariance)
        params.dt             # I (identity matrix)
    )
    state = [1.0;0;0;0;0;0;0]
    # state_estimation_history =  Matrix{Float64}(undef, 7, size(mag_noisy_history)[2])
    P = 1.0 * Matrix{Float64}(I, 6, 6)
    # for i in 1:size(mag_noisy_history)[2]
    #     state, P = update(state, P, kf, (mag_noisy[:,i],sun_noisy[:,i]), (mag_eci[i],sun_eci[i]))
    #     state, P = predict(state, P, kf, gyroscope_measurement[:,i])
    #     state_estimation_history[:,i] = state
    #     # state_estimation_history = hcat(state_estimation_history, state)
    # end
    # return state_estimation_history

    # Create an empty list to hold state estimations
    state_estimation_list = []

    # Main loop
    for i in 1:size(mag_noisy_history)[2]
        state, P = update(state, P, kf, (mag_noisy[:,i], sun_noisy[:,i]), (mag_eci[i], sun_eci[i]))
        state, P = predict(state, P, kf, gyroscope_measurement[:,i])
        
        # Append the current state to the list
        ignore() do 
            push!(state_estimation_list, state)
        end
    end

    # Convert the list to a matrix
    state_estimation_history = hcat(state_estimation_list...)
end
