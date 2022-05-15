import pandas as pd
import proplot as pplt
import seaborn as sns


def plot_mode(mode, path, ax):
    csvs = []
    for csv in path.glob(f"*{mode}.csv"):
        data = pd.read_csv(csv)
        if csv.name.startswith("cifar10_baseline"):
            data["run"] = "Baseline"
        elif csv.name.startswith("cifar10_trivialaugment"):
            data["run"] = "TrivialAugment"
        else:
            raise NotImplementedError
        csvs.append(data)
    data = pd.concat(csvs, ignore_index=True)
    style = sns.axes_style(style="ticks", rc={})
    sns.set_theme(style=style)
    plt = sns.lineplot(x="Step", y="Value", data=data, hue="run", ax=ax)
    plt.legend([], [], frameon=False)  # Remove seaborn legend as we use the one from proplot


if __name__ == "__main__":
    from pathlib import Path

    path = Path("metassl/analysis/plot/tensorboard_csv")

    fig, axs = pplt.subplots(ncols=2, nrows=1, share=False)
    modes = ["test-acc", "pt-loss"]
    for ax, mode in zip(axs, modes):
        plot_mode(mode, path, ax)

    # Organize axs and legend(s)
    axs[0].format(
        ylabel="Test Accuracy",
        xlabel="#Steps",
    )
    axs[1].format(
        ylabel="Pre-training Loss",
        xlabel="#Steps",
    )
    axs[0].legend(center=True, frame=True, loc="b")
    axs.format(
        suptitle="Learning Curves CIFAR-10 SimSiam",
    )

    # Save plot
    fig.savefig("metassl.pdf")
