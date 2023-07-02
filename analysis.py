import matplotlib.pyplot as plt
import pandas as pd
import pickle
import adjust

model_name = adjust.model_name
# model_name="fastrrcnn4"

line_style = ["solid", "dotted", "dashed", "dashdot"]


def plot_losses_for_one_nettype_compare(net_name, losses_list):
    fig, ax = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Анализ сети " + net_name)
    ax[0][0].set_title("losses по выборке train")
    ax[0][1].set_title(" IoU score по выборке train")
    ax[1][0].set_title("losses по выборке test")
    ax[1][1].set_title(" IoU score по выборке test")
    ax[2][0].set_title("losses по выборке val")
    ax[2][1].set_title(" IoU score по выборке val")

    for i in range(len(losses_list)):
        df = pd.read_csv(net_name + "_" + losses_list[i] + ".csv")

        ax[0][0].plot(df["loss_train"], label=str(losses_list[i]), c="#12DDDD", linestyle=line_style[i])
        ax[1][0].plot(df["loss_test"], label=losses_list[i], c="#12DD12", linestyle=line_style[i])
        ax[2][0].plot(df["loss_val"], label=losses_list[i], c="#991122", linestyle=line_style[i])

        ax[0][1].plot(df["score_train"], label=str(losses_list[i]) + str(" {:.3f}".format(df["score_train"].max())),
                      c="#12DDDD", linestyle=line_style[i])
        ax[1][1].plot(df["score_test"], label=str(losses_list[i]) + str(" {:.3f}".format(df["score_test"].max())),
                      c="#12DD12", linestyle=line_style[i])
        ax[2][1].plot(df["score_val"], label=str(losses_list[i]) + str(" {:.3f}".format(df["score_val"].max())),
                      c="#991122", linestyle=line_style[i])

    [ax[i][j].legend() for i in range(3) for j in range(2)]
    [ax[i][1].set_ylim(0, 1) for i in range(3)]

    plt.show()
def plot_losses(model_name):
    df = pd.read_csv(model_name+".csv")
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.plot(df["losses_train"])
    ax.plot(df["losses_test"])
    plt.show()

plot_losses(model_name)

#plot_losses_for_one_nettype_compare("fastrrcnn4", pd.DataFrame(analysis_csv))