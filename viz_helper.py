# plotting stuff
from IPython.display import clear_output
import matplotlib

matplotlib.rc("font", size=16)
matplotlib.rc("lines", linewidth=2.5)
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import jax.numpy as jnp


def barplot_color(x, y):
    import matplotlib as mpl

    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(x)))
    plt.bar(x, y, color=colors)
    plt.xticks(x, labels=x)


def plt_save_fig_array(close=True, clear=True):
    fig = plt.gcf()
    fig.canvas.draw()
    res = np.array(fig.canvas.renderer.buffer_rgba())
    if close:
        plt.close()
    if clear:
        plt.clf()
    return res


def animate(clip, filename="animation.mp4", _return=True, fps=10, embed=False):
    # embed = True for Pycharm, otherwise False
    if isinstance(clip, dict):
        clip = clip["image"]
    print(f"animating {filename}")
    if filename.endswith(".gif"):
        import imageio
        import matplotlib.image as mpimg

        imageio.mimsave(filename, clip)
        if _return:
            from IPython.display import display
            import ipywidgets as widgets

            return display(
                widgets.HTML(f'<img src="{filename}" width="750" align="center">')
            )
        else:
            return

    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(clip, fps=fps)
    ftype = filename[-3:]
    if ftype == "mp4":
        clip.write_videofile(filename, fps=fps)
    elif ftype == "gif":
        clip.write_gif(filename, fps=fps)
    else:
        raise NotImplementedError(f"file type {ftype} not supported!")

    if _return:
        from IPython.display import Video

        return Video(
            filename, embed=embed, html_attributes="controls autoplay muted loop"
        )


# visualize the distribution of actions
def dist_hist(actions, bins):
    x_freq = np.linspace(-2.5, 2.5, bins + 1)
    inds = np.digitize(actions, x_freq)
    freq = np.zeros_like(x_freq)
    unique, counts = np.unique(inds, return_counts=True)
    for i, count in zip(unique, counts):
        freq[i - 1] = count
    freq = freq / freq.sum() * 2
    return x_freq + (x_freq[1] - x_freq[0]) / 2, freq * 2.4


def visualize_policy(actions, pdf_scale=1):
    plt.scatter(actions, np.zeros_like(actions) - 0.05, s=5, alpha=0.2)
    plt.clim(actions.min(), actions.max())
    x_freq, freq = dist_hist(actions, bins=50)
    X_Y_Spline = make_interp_spline(x_freq, freq)
    X_ = np.linspace(x_freq.min(), x_freq.max(), 1000)
    Y_ = np.clip(X_Y_Spline(X_), a_min=0, a_max=None)
    # spline interpolates out of the action bound, it looks confusing
    Y_[np.abs(X_) > 1] = 0
    Y_ *= pdf_scale
    plt.plot(X_, Y_, color="C3", linewidth=2)
    plt.fill_between(X_, Y_, color="C3", alpha=0.2)


def plot_reward(x, y):
    plt.xlim(-1.2, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.plot(x.reshape(-1), y.reshape(-1), color="green")


def make_plot(
    actions,
    reward_fn,
    rewards,
    gm_loss,
    policy_loss,
    visualize_every,
    total_steps,
    label="Diffusion",
):
    total_plots = 4
    fig_width = 4

    plt.figure(figsize=(total_plots * fig_width, fig_width))

    plt.subplot(1, total_plots, 1)
    plt.title(label)
    x = jnp.linspace(-1.5, 1.5, 200)
    y = reward_fn(x)
    plot_reward(x, y)
    visualize_policy(actions)

    # visualize reward
    plt.subplot(1, total_plots, 2)
    plt.xlim(0, total_steps)
    plt.ylim(-0.1, 1.1)
    plt.title("Reward")
    plt.plot(
        np.arange(len(rewards)) * visualize_every,
        rewards,
        color="C0",
        label=label,
    )
    plt.grid()
    plt.legend(loc="lower right")

    # visualize ddpm loss
    plt.subplot(1, total_plots, 3)
    plt.xlim(0, total_steps)
    plt.ylim(-0.1, 1.5)
    plt.title("GM Loss")
    plt.plot(np.arange(len(gm_loss)) * visualize_every, gm_loss, label="gm_loss")
    plt.grid()
    plt.legend(loc="upper right")

    # visualize policy loss
    plt.subplot(1, total_plots, 4)
    plt.xlim(0, total_steps)
    plt.ylim(-0.1, 0.5)
    plt.title("Policy Loss")
    plt.plot(
        np.arange(len(policy_loss)) * visualize_every, policy_loss, label="policy_loss"
    )
    plt.grid()
    plt.legend(loc="upper right")

    plt.tight_layout()
    fig_array = plt_save_fig_array(close=False, clear=False)
    # plt.show()
    return fig_array
