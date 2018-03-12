import numpy as np

vis = None

win = None

avg_reward = 0

X = []
Y = []

def visdom_plot(
    total_num_steps,
    mean_reward
):
    # Lazily import visdom so that people don't need to install visdom
    # if they're not actually using it
    from visdom import Visdom

    global vis
    global win
    global avg_reward

    if vis is None:
        vis = Visdom()
        assert vis.check_connection()

        # Close all existing plots
        vis.close()

    # Running average for curve smoothing
    avg_reward = avg_reward * 0.9 + 0.1 * mean_reward

    X.append(total_num_steps)
    Y.append(avg_reward)

    # The plot with the handle 'win' is updated each time this is called
    win = vis.line(
        X = np.array(X),
        Y = np.array(Y),
        opts = dict(
            #title = 'All Environments',
            xlabel='Total time steps',
            ylabel='Reward per episode',
            ytickmin=0,
            #ytickmax=1,
            #ytickstep=0.1,
            #legend=legend,
            #showlegend=True,
            width=900,
            height=500
        ),
        win = win
    )
