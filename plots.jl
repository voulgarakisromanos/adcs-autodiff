using Plots

function plot_histories(groundtruth_state_history, state_estimation_history)
    # Initialize a plot with 7 subplots arranged in a 7x1 grid
    plot(layout = (7, 1), size = (600, 1200))

    # Loop through each row of the matrices and plot them in the same subplot
    for i in 1:7
        plot!(subplot = i, groundtruth_state_history[i, :], label = "Ground Truth", xlabel = "Timestep", ylabel = "Value", linewidth = 2)
        plot!(subplot = i, state_estimation_history[i, :], label = "Estimated", xlabel = "Timestep", ylabel = "Value", linewidth = 2)
    end

    # Show the plot
    display(current())
end