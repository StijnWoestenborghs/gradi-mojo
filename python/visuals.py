import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_gradient_descent_2D(positions_over_time, loss_over_time, title = 'Gradient descent'):
    print(f"Plotting: {title}...")
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


def flatten(points):
    dim = points.shape[2]

    x = points.reshape(-1, dim)[:, 0]
    y = points.reshape(-1, dim)[:, 1]
    
    if dim == 2:
        # If the data is 2D, set z-coordinates to zero
        z = np.zeros_like(x)
    elif dim == 3:
        z = points.reshape(-1, dim)[:, 2]
    else:
        raise ValueError("Only 2D and 3D supported")
    
    return x, y, z


def plot_gradient_descent(positions_over_time, loss_over_time, title = 'Gradient descent'):
    print(f"Plotting: {title}...")
    points = np.array(positions_over_time)
    if points.ndim == 2:
        points = points[np.newaxis, :, :]
    if not isinstance(loss_over_time, (list, np.ndarray)):
        loss_over_time = [loss_over_time]

    N_points = len(points[0])
    time_steps = len(points)

    x, y, z = flatten(points)

    # Create a figure
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(color=['gray'] * (N_points * (time_steps - 1)) + ['red'] * N_points, 
                        size=[5] * (N_points * (time_steps - 1)) + [15] * N_points,
                        line=dict(width=0))
        )],
        layout=go.Layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[-2, 2], autorange=False),
                yaxis=dict(range=[-2, 2], autorange=False),
                zaxis=dict(range=[-2, 2], autorange=False),
                aspectmode='cube'
            ),
            annotations=[dict(
                showarrow=False, x=0.90, y=0.90,
                xref="paper", yref="paper",
                text=f"Iteration: {len(loss_over_time)-1}<br>Loss: {loss_over_time[-1]:.4f}",
                font=dict(size=25)
            )]
        )
    )

    fig.show()


def animate_gradient_descent(positions_over_time, loss_over_time, title="Gradient Descent Animation", trace=False):
    print(f"Animating: {title}...")
    points = np.array(positions_over_time)
    if points.ndim == 2:
        points = points[np.newaxis, :, :]

    N_points = len(points[0])
    time_steps = len(points)

    all_x, all_y, all_z = flatten(points)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=all_x[:N_points], 
            y=all_y[:N_points], 
            z=all_z[:N_points], 
            mode='markers', 
            marker=dict(color=["red"] * N_points, size=10)
        )],
        layout=go.Layout(
            title=title,
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(frame=dict(duration=1, redraw=True), fromcurrent=True, mode="immediate")]
                    )
                    ]
                )],
            scene=dict(
                xaxis=dict(range=[-2, 2], autorange=False),
                yaxis=dict(range=[-2, 2], autorange=False),
                zaxis=dict(range=[-2, 2], autorange=False),
                aspectmode='cube'
            )
        ),
        frames=[go.Frame(
            data=[
                go.Scatter3d(
                    x=all_x[:(i+1)*N_points] if trace else all_x[i*N_points:(i+1)*N_points],
                    y=all_y[:(i+1)*N_points] if trace else all_y[i*N_points:(i+1)*N_points],
                    z=all_z[:(i+1)*N_points] if trace else all_z[i*N_points:(i+1)*N_points],
                    mode='markers',
                    marker=dict(
                        color=['gray']*(N_points * i) + ['red']*N_points if trace else ['red']*N_points, 
                        size=[5]*(N_points * i) + [15]*N_points if trace else [15]*N_points,
                        line=dict(width=0) 
                    )
                )
            ],
            layout=dict(annotations=[dict(
                showarrow=False, x=0.90, y=0.90,
                xref="paper", yref="paper",
                text=f"Iteration: {i+1}<br>Loss: {loss_over_time[i]:.4f}",
                font=dict(size=25)
            )])
            ) for i in range(1, time_steps)]
    )

    fig.show()
