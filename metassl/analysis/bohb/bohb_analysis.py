# Code based on: https://automl.github.io/HpBandSter/build/html/auto_examples/plot_example_6_analysis.html
import os
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt

def make_analysis(path):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(path)

    ######################################################################################
    # RESULTS
    ######################################################################################
    # For more see:
    # https://github.com/automl/HpBandSter/blob/master/hpbandster/core/result.py

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()
    print("RESULT", result)

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    print("HERE:", id2conf, inc_id)
    inc_config = id2conf[inc_id]["config"]

    print("Best found configuration:")
    print(inc_config)

    # Save best configuration
    file_to_write_in = open(path + "/results_loss_and_info.txt", "w")
    file_to_write_in.write("Best found configuration:" + "\n" + str(inc_config) + "\n\n")
    file_to_write_in.close()

    ######################################################################################
    # PLOTS
    ######################################################################################

    plots_folder = os.mkdir(path + "/bohb_plots")  # noqa: F841
    path_plots = path + "/bohb_plots"

    # Observed losses grouped by budget
    hpvis.losses_over_time(all_runs)
    plt.savefig(path_plots + "/losses_over_time.png")

    hpvis.concurrent_runs_over_time(all_runs)
    plt.savefig(path_plots + "/concurrent_runs_over_time.png")

    hpvis.finished_runs_over_time(all_runs)
    plt.savefig(path_plots + "/finished_runs_over_time.png")

    # Plot visualizes the spearman rank correlation coefficients of the losses between
    # different budgets.
    hpvis.correlation_across_budgets(result)
    plt.savefig(path_plots + "/correlation_across_budgets.png")

    # For model based optimizers, one might wonder how much the model actually helped.
    # Plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    plt.savefig(path_plots + "/performance_histogram_model_vs_random.png")


# if __name__ == "__main__":
#     result_path = "/home/wagn3rd/Projects/metassl/results/diane/BO_color-jitter-strength_new-val"
#     make_analysis(path=result_path)

# https://github.com/automl/HpBandSter/blob/master/hpbandster/core/result.py



def get_pandas_dataframe(result, budgets=None, loss_fn=lambda r: r.loss):
    import numpy as np
    import pandas as pd

    id2conf = result.get_id2config_mapping()

    df_x = pd.DataFrame()
    df_y = pd.DataFrame()

    if budgets is None:
        budgets = result.HB_config['budgets']

    all_runs = result.get_all_runs(only_largest_budget=False)
    all_runs = list(filter(lambda r: r.budget in budgets, all_runs))

    all_configs = []
    all_losses = []

    for r in all_runs:
        if r.loss is None: continue
        config = id2conf[r.config_id]['config']
        if len(budgets) > 1:
            config['budget'] = r.budget

        all_configs.append(config)
        all_losses.append({'loss': loss_fn(r)})

    # df_x = df_x.append(config, ignore_index=True)
    # df_y = df_y.append({'loss': r.loss}, ignore_index=True)

    df_X = pd.DataFrame(all_configs)
    df_y = pd.DataFrame(all_losses)

    return (df_X, df_y)

# for validation
def get_incumbent_value(metrics, iterations):
    return max(metrics[:iterations+1])

# for test
def get_best_validation_index(metrics, iterations):
    values = metrics[:iterations+1]
    return max(range(len(values)), key=values.__getitem__)

# %%
result_path = "/home/wagn3rd/Projects/metassl/results/diane/BO_color-jitter-strength_new-val"
result = hpres.logged_results_to_HBS_result(result_path)

configs, val_metrics = result.get_pandas_dataframe()
val_metrics = -val_metrics
val_metrics = val_metrics.to_dict()["loss"]
val_metrics = [val_metrics[i] for i in range(len(val_metrics))]

#%%
configs, test_metrics = get_pandas_dataframe(result, budgets=None, loss_fn=lambda r: r.info["test/metric"])
test_metrics = test_metrics.to_dict()["loss"]
test_metrics = [test_metrics[i] for i in range(len(test_metrics))]




#%%
test_incumbent = [get_best_validation_index(val_metrics, i) for i in range(len(test_metrics))]
y = get_best_validation_index(test_metrics, 1)

#%%
test_incumbent = [test_metrics[i] for i in test_incumbent]
val_incumbent = [get_incumbent_value(val_metrics, i) for i in range(len(val_metrics))]

# for plot 1
baseline_val = [91.41, 91.67, 91.45, 92.19, 91.49]
baseline_test = [90.47, 91.14, 90.80, 90.91, 90.74]

# for plot 2
best_baseline_val = [92.19]
best_baseline_test = [90.91]

#%%
# for plot 1
baseline_records_val = []
for i in range(1, len(val_incumbent) + 1):
    for baseline_acc in baseline_val:
        baseline_records_val.append(dict(Accuracy=baseline_acc, Run="baseline_val", Iteration=i))

baseline_records_test = []
for i in range(1, len(test_incumbent) + 1):
    for baseline_acc in baseline_test:
        baseline_records_test.append(dict(Accuracy=baseline_acc, Run="baseline_val_on_test", Iteration=i))

# for plot 2
best_baseline_records_val = []
for i in range(1, len(val_incumbent) + 1):
    for baseline_acc in best_baseline_val:
        best_baseline_records_val.append(dict(Accuracy=baseline_acc, Run="best_baseline_val", Iteration=i))

best_baseline_records_test = []
for i in range(1, len(test_incumbent) + 1):
    for baseline_acc in best_baseline_test:
        best_baseline_records_test.append(dict(Accuracy=baseline_acc, Run="best_baseline_val_on_test", Iteration=i))

#%%
# BOHB
test_records = [dict(Accuracy=acc, Run="best_bohb_val_on_test", Iteration=i) for i, acc in enumerate(test_incumbent, 1)]
val_records = [dict(Accuracy=acc, Run="best_bohb_val", Iteration=i) for i, acc in enumerate(val_incumbent, 1)]


#%%
import pandas as pd
# for plot 1
data = pd.DataFrame.from_records(val_records + test_records + baseline_records_val + baseline_records_test)

# for plot 2
best_data = pd.DataFrame.from_records(val_records + test_records + best_baseline_records_val + best_baseline_records_test)

#%%
import seaborn as sns
# sns.set_context("talk")
sns.set_style("ticks")  # todo: check

#%%
plot_1 = sns.lineplot(data=data, x="Iteration", y="Accuracy", hue="Run", ci="sd")  # 66: 66% confidence intervall > means: standard error
figure_1 = plot_1.get_figure()
figure_1.show()


#%%
plot_2 = sns.lineplot(data=best_data, x="Iteration", y="Accuracy", hue="Run", ci="sd")
figure_2 = plot_2.get_figure()
figure_2.show()