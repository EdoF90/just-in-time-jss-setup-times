# -*- coding: utf-8 -*-
import os
from instances import *
from pandas import DataFrame
import matplotlib.pyplot as plt
from sol_representation import *
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def plot_gantt_chart(df_sol: DataFrame, setups: dict, inst: Instance, add_info=True, file_path=None, plot_legend=False, messy_sol=False):
    t_end_max = df_sol.t_end.max()
    HEIGHT_LINE = 0.6 * t_end_max / inst.n_machines / 5
    Y_MAX = t_end_max / inst.n_machines / 5
    def get_y_coord(machine):
        return inst.lst_machines.index(machine) * Y_MAX
    fig, ax = plt.subplots(1, 1,figsize=(20,10))
    ax.set_yticks(
        [get_y_coord(machine) for machine in inst.lst_machines]
    )
    labels = [f"{ele}" for ele in inst.lst_machines]
    ax.set_yticklabels(
        labels,
        rotation='horizontal',
        fontsize=15
    )
    # assign to each job a color:
    job_lst = [ele for ele in df_sol.order.unique()]
    job_lst.sort()
    job_colors = get_random_colors_dict(
        job_lst, rgb=True
    )
    # PLOT TASKS
    for _, task in df_sol.iterrows():
        # Plot rectangle
        y_coord = get_y_coord(task['machine'])
        if 'split_ref' in task:
            if task.split_ref == -1:
                # it is not a splitted task:
                rectangle = plt.Rectangle(
                    xy=(task['t_start'], y_coord - HEIGHT_LINE / 2),
                    width=task['t_end'] - task['t_start'],
                    height=HEIGHT_LINE,
                    fc=job_colors[task['op'][1]],
                    ec="red"
                )
            else:
                color = job_colors[task.split_ref]
                rectangle = plt.Rectangle(
                    xy=(task['t_start'], y_coord - HEIGHT_LINE / 2),
                    width=task['t_end'] - task['t_start'],
                    height=HEIGHT_LINE,
                    fc=(color[0], color[1], color[2], 0.5),
                    ec=color,
                    hatch="/"
                )
        else:
            rectangle = plt.Rectangle(
                xy=(task['t_start'], y_coord - HEIGHT_LINE / 2),
                width=task['t_end'] - task['t_start'],
                height=HEIGHT_LINE,
                fc=job_colors[task['op'][1]],
                ec="red"
            )
        fig.gca().add_patch(rectangle)
        rx, ry = rectangle.get_xy()
        cx = rx + rectangle.get_width()/2.0
        cy = ry + rectangle.get_height()/2.0
        if add_info:
            if 'split_ref' in task and task.split_ref != -1:
                str_annotation = f"{task['op']} [{task.split_ref}]\n[{task['t_start']:.1f}-{task['t_end']:.1f}]"
            else:
                str_annotation = f"{task['op']}\n[{task['t_start']:.1f}-{task['t_end']:.1f}]"
            plt.annotate(
                str_annotation,
                (cx, cy), color='black', weight='bold',
                fontsize=10, ha='center', va='center'
            )
        if messy_sol:
            # print info on the screen (useful for messy solutions)
            print(f"ord{task['op'][1]}- {task['op'][0]} - [{task['t_start']:.1f}-{task['t_end']:.1f}]")

    # PLOT SETUP
    for setup in setups:
        # Plot rectangle
        y_coord = get_y_coord(setup['machine'])
        rectangle = plt.Rectangle(
            xy=(setup['t_start'], y_coord - HEIGHT_LINE / 2),
            width=setup['t_end'] - setup['t_start'],
            height=HEIGHT_LINE,
            fc="gray",
            alpha=0.5,
            ec="black"
        )
        fig.gca().add_patch(rectangle)

    # PLOT VERTICAL LINES AND LEGEND
    # Legend initialization
    legend_elements = []
    legend_elements.append(
        mpatches.Patch(
            fc="gray",
            alpha=0.5,
            ec="black",
            label='Setup'
        )
    )

    for idx_po, po in inst.jobs.items():
        # Adding release date in the legend
        if plot_legend:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=job_colors[idx_po],
                    lw=2,
                    label=f"{po['item_name']}-{idx_po} release date",
                    linestyle='--'
                ),
            )
            # Adding due date in the legend
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=job_colors[idx_po],
                    lw=2,
                    label=f"{po['item_name']}-{idx_po} due date"
                )
            )
        # Adding release date line in the graph
        # 1.get machine
        tmp = df_sol[(df_sol.order == idx_po)]
        tmp = tmp[tmp.t_start == tmp.t_start.min()]
        machine = tmp.machine.iloc[0]
        y_center = get_y_coord(machine)
        plt.vlines(
            x=po['release_date'],
            ymin= y_center - HEIGHT_LINE * 0.75,
            ymax= y_center + HEIGHT_LINE * 0.75,
            colors=job_colors[idx_po],
            linestyles='dashed'
        )
        # Adding deadline line in the graph
        plt.vlines(
            x=po['due_date'],
            ymin=-HEIGHT_LINE,
            ymax=len(inst.lst_machines) * Y_MAX,
            colors=job_colors[idx_po]
        )
    # Create the figure
    ax.legend(
        handles=legend_elements,
        loc='upper center', bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, ncol=1
    )
    plt.ylim(bottom=-1, top=len(inst.lst_machines) + 1)
    plt.tight_layout()
    plt.grid()
    plt.axis('scaled')
    if file_path is None:
        plt.show()
    else:
        plt.savefig(
            os.path.join('.', 'results', file_path),
            bbox_inches="tight"
        )
    plt.close(fig)
