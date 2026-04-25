import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import RegistrationLibrary_Melara_Unai as RegistrationLibrary

def plot_inputs(target, source):
    plt.figure(figsize=(8, 6))
    plt.scatter(source[:, 0], source[:, 1], c='blue', label='Source', alpha=0.6, marker='o')
    plt.scatter(target[:, 0], target[:, 1], c='red', label='Target', alpha=0.6, marker='o')

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Input PointClouds")
    plt.axis("equal")
    plt.show()

def transform_points(points, transformation):
    dim = points.shape[1]
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = homogeneous_points @ transformation.T
    return transformed_points[:, :dim]

def generate_registration_animation(target, source, history):
    fig, ax = plt.subplots(1, 1)
    ax.set_title("ICP Iterations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    margin_factor = 0.5
    
    # Calculate the min and max of source and target points
    x_min = min(np.min(source[:, 0]), np.min(target[:, 0]))
    x_max = max(np.max(source[:, 0]), np.max(target[:, 0]))
    y_min = min(np.min(source[:, 1]), np.min(target[:, 1]))
    y_max = max(np.max(source[:, 1]), np.max(target[:, 1]))
    
    # Add margin to the x and y limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Apply margin to x and y limits
    ax.set_xlim([x_min - margin_factor * x_range, x_max + margin_factor * x_range])
    ax.set_ylim([y_min - margin_factor * y_range, y_max + margin_factor * y_range])

    # Force equal aspect ratio:
    ax.set_aspect('equal')
    
    # Plot target:
    ax.scatter(target[:, 0], target[:, 1],
               s=4, color="red", label="Target Points")

    # Animated plot source:
    animated_plot, = ax.plot(source[:, 0], source[:, 1],
                             'bo',
                             markersize=2, 
                             label="Transformed Source Points")
    
    # Text box:
    info_box = ax.text(0.05, 0.95, '',
                       transform=ax.transAxes, fontsize=12,
                       verticalalignment='top',
                       bbox=dict(facecolor='white', alpha=0.5))


    def update(i):
        if i != 0:
            metric = history[i-1][0]
            info_box.set_text(f"Iteration: {i}\nRMSE: {metric:.4f}")
    
            transformation = history[i-1][1]  
            transformed_points = transform_points(source, transformation)
            animated_plot.set_data(transformed_points[:, 0], transformed_points[:, 1])
        return animated_plot, info_box
    ani = animation.FuncAnimation(fig, update, frames=len(history)+1,  interval=500)
    plt.legend(loc="lower center")
    
    ani.save("Registration2DAnimation.gif", writer='pillow')
    #ani.save("Registration2DAnimation.mp4")

def generate_damping_sinusoidal_points(
        amplitude,
        frequency,
        phase,
        num_points,
        x_range,
        damping_factor
        ):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    damped_amplitude = amplitude * np.exp(-damping_factor * (x_values - x_range[0]))
    y_values = damped_amplitude * np.sin(frequency * x_values + phase)
    return np.column_stack((x_values, y_values)) 

def generate_2D_transformation(angle, t_x, t_y):
    pose = np.array([[np.cos(angle), -np.sin(angle), t_x],
                     [np.sin(angle), np.cos(angle), t_y],
                     [0, 0, 1]])
    return pose

def add_gaussian_noise(points, sigma):
    noise = np.random.normal(loc=0, scale=sigma, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def add_point_to_points(points, point):
    return np.vstack((points, point))

def main():
    # Point generation:
    np.random.seed(42)
    target = generate_damping_sinusoidal_points(50, 0.08, 0, 100, [-50,50], 0.02)
    
    T = generate_2D_transformation(np.pi /4, 1, -2)
    source = transform_points(target, T)
    
    # Perturbate source:
    source = add_gaussian_noise(source, 2)
    
    source = add_point_to_points(source, (-30,-60))
    source = add_point_to_points(source, (-20,-6.50))
    source = add_point_to_points(source, (-20,-5.50))
    source = add_point_to_points(source, (-20.5,-50))
    source = add_point_to_points(source, (-20.5,-50.5))
    source = add_point_to_points(source, (-20,-5))
    source = add_point_to_points(source, (-20.5,-60.5))
    source = add_point_to_points(source, (-20,-60.5))
    source = add_point_to_points(source, (-20.5,-60))
    
    #n_outliers = 1000
    #angles = np.random.uniform(0, 2 * np.pi, n_outliers)
    #radii = np.random.uniform(60, 120, n_outliers)
    #outliers = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    #source = np.vstack((source, outliers))


    # Plot inputs
    plot_inputs(target, source)
    
    # Execute ICP registration:
    T_est, history = RegistrationLibrary.icp(target, source)
    
    # Generate an animation:
    generate_registration_animation(target, source, history)
    print("Matriz de transformación aplicada:\n", T)
    print("Matriz de transformación estimada:")
    print(np.linalg.inv(T_est))

if __name__ == "__main__":
    main()
