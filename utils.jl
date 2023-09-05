using LinearAlgebra  # For matrix multiplication
using SatelliteDynamics
using Statistics

function quat_mult(q, p)
    return [
        q[1]*p[1] - q[2]*p[2] - q[3]*p[3] - q[4]*p[4],
        q[1]*p[2] + q[2]*p[1] + q[3]*p[4] - q[4]*p[3],
        q[1]*p[3] - q[2]*p[4] + q[3]*p[1] + q[4]*p[2],
        q[1]*p[4] + q[2]*p[3] - q[3]*p[2] + q[4]*p[1]
    ]
end

function quaternion_exp(q::Vector{Float64})
    w, v = q[1], q[2:4]
    v_norm = norm(v)
    
    if v_norm == 0.0
        return [exp(w), 0.0, 0.0, 0.0]
    end
    
    half_angle = v_norm / 2
    scalar_part = cos(half_angle)
    vector_part = (v / v_norm) * sin(half_angle)
    
    return [scalar_part; vector_part]
end

function rotate_vector_by_quaternion(v::AbstractArray, q::Vector{Float64})
    # Extract the components of the quaternion
    w, u = q[1], [q[2], q[3], q[4]]
    # Rotate the vector
    v_rotated = v + 2.0 * cross(u, cross(u, v) + w .* v)
    return v_rotated
end

function skew_symmetric(v::AbstractArray)
    x, y, z = v
    S = [ 0.0 -z   y;
          z   0.0 -x;
         -y   x   0.0]
    return S
end
function julian_to_gregorian(jd::Real)
    Z = floor(jd + 0.5)
    F = (jd + 0.5) - Z
    A = Z
    if Z >= 2299161
        alpha = floor((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - floor(alpha / 4)
    end
    B = A + 1524
    C = floor((B - 122.1) / 365.25)
    D = floor(365.25 * C)
    E = floor((B - D) / 30.6001)
    
    day = B - D - floor(30.6001 * E) + F
    month = (E < 14) ? E - 1 : E - 13
    year = (month > 2) ? C - 4716 : C - 4715
    
    # Calculate time components
    F_day = day - floor(day)
    hour = floor(F_day * 24)
    minute = floor((F_day * 24 - hour) * 60)
    second = (F_day * 24 - hour - minute / 60) * 3600
    microsecond = (second - floor(second)) * 1e9  # Convert remaining fraction to nanoseconds
    
    return Int(year), Int(month), Int(floor(day)), Int(hour), Int(minute), Float64(floor(second)), Float64(floor(microsecond))
end