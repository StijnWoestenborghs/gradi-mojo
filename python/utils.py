import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_gradient_descent(positions_over_time, loss_over_time, title = 'Gradient descent'):
    positions_over_time = np.array(positions_over_time)
    X_final = positions_over_time[-1]
    loss_final = loss_over_time[-1]

    fig, ax = plt.subplots()
    
    for positions in positions_over_time:
        ax.scatter(positions[:, 0], positions[:, 1], c='gray', alpha=0.5)
    ax.scatter(X_final[:, 0], X_final[:, 1], c='red')

    text = f"Iteration: {len(loss_over_time)-1}\nLoss: {loss_final:.4f}"
    ax.text(0.98, 0.90, text, transform=ax.transAxes, ha='right', fontsize=10)
    plt.axis('equal')
    plt.title(title)
    plt.show()


def animate_gradient_descent(positions_over_time, loss_over_time):
    positions_over_time = np.array(positions_over_time)
    X_final = positions_over_time[-1]
    loss_final = loss_over_time[-1]

    # Animation
    fig, ax = plt.subplots()
        
    trace_scat = ax.scatter([], [], s=10, c='gray', alpha=0.5)
    current_scat = ax.scatter(X_final[:, 0], X_final[:, 1], c='red')
    all_previous_positions = []

    text = ax.text(0.98, 0.90, "", transform=ax.transAxes, ha='right', fontsize=10)

    def update(frame):
        positions = positions_over_time[frame]
        all_previous_positions.extend(positions.tolist())
        trace_scat.set_offsets(all_previous_positions)
        current_scat.set_offsets(positions)

        # Update the text
        current_loss = loss_over_time[frame]
        text.set_text(f"Iteration: {frame}\nLoss: {current_loss:.4f}")
        
        return trace_scat, current_scat, text

    ani = FuncAnimation(fig, update, frames=len(loss_over_time), blit=True)

    plt.axis('equal')
    plt.show()
