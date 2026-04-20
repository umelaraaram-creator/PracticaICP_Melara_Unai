import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import RegistrationLibrary_Melara_Unai as RegistrationLibrary
import trimesh
import time

def load_mesh(path):
    return trimesh.load_mesh(path)

def sample_points_from_mesh(mesh, number_points):
    points, _ = trimesh.sample.sample_surface_even(mesh, number_points)
    return points

def plot_inputs_3D(target, source):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        target[:, 0], target[:, 1], target[:, 2],
        c='red', label='Target', alpha=0.6, marker='.'
        )
    ax.scatter(
        source[:, 0], source[:, 1], source[:, 2],
        c='blue', label='Source', alpha=0.6, marker='.'
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Input 3D Point Clouds")
    ax.legend()
    plt.show()

def generate_3D_transformation(angle_x, angle_y, angle_z, t_x, t_y, t_z):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t_x, t_y, t_z]

    return T


def transform_points(points, transformation):
    dim = points.shape[1]
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = homogeneous_points @ transformation.T
    return transformed_points[:, :dim]


def generate_registration_animation_3D(target, source, history):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title("ICP 3D Iterations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    ax.scatter(
        target[:, 0], target[:, 1], target[:, 2],
        c='red', label='Target', alpha=0.6, marker='.'
        )
    animated_plot = ax.scatter(
        source[:, 0], source[:, 1], source[:, 2],
        c='blue', label='Transformed Source', alpha=0.6, marker='.'
        )
    
    info_box = ax.text2D(
        0.05, 0.95, '', transform=ax.transAxes,
        fontsize=12, bbox=dict(facecolor='white', alpha=0.5)
        )
    
    def update(i):
        if i != 0:
            metric = history[i - 1][0]
            info_box.set_text(f"Iteration: {i}\nRMSE: {metric:.4f}")
            
            transformation = history[i - 1][1]
            transformed_points = transform_points(source, transformation)
            animated_plot._offsets3d = (
                transformed_points[:, 0],
                transformed_points[:, 1],
                transformed_points[:, 2]
                )
        
        return animated_plot, info_box

    ani = animation.FuncAnimation(
        fig, update, frames=len(history) + 1, interval=500
        )
    plt.legend()
    
    ani.save("Registration2DAnimation.gif", writer='pillow')
    #ani.save("Registration3DAnimation.mp4")

def add_gaussian_noise_3D(points, sigma):
    noise = np.random.normal(0, sigma, points.shape)
    return points + noise

def add_point_to_points(points, point):
    return np.vstack((points, point))

def main():
    np.random.seed(42)
    
    # Load target model:
    path = "rabbit.ply"
    target_mesh = load_mesh(path)
    target = sample_points_from_mesh(target_mesh, 3000)
    
    # Generate source by transforming target    
    T = generate_3D_transformation( 0, np.pi / 8, np.pi / 16,  0.1,  0.1, 0.1)
    source = transform_points(target, T)
    source = add_gaussian_noise_3D(source, 0.0)
    
    plot_inputs_3D(target, source)
    
    # Run ICP
    t = time.process_time()
    T_est, history = RegistrationLibrary.icp(target, source)
    elapsed_time = time.process_time() - t
    
    
    # Generate animation
    generate_registration_animation_3D(target, source, history)
    print("Registration completed in ", elapsed_time, " seconds.")
    print("Applied Transformation Matrix:\n", T)

    print("Estimated Transformation Matrix:")
    print(np.linalg.inv(T_est))

if __name__ == "__main__":
    main()
