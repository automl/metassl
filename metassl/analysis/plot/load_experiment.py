import numpy as np


def load_experiment(expt_dict, keys):

    summary_dict = np.load(expt_dict["dir"] / "summary_dict.npy", allow_pickle=True).tolist()
    try:
        file = open(expt_dict["dir"] / "log_file.txt")
        test_dict = {
            line.split(" ")[1].split(":")[0]: float(line.split(" ")[2].replace(",", "."))
            for line in file
            if "_test" in line
        }
        print("load test dict")
    except Exception:
        test_dict = {}
        print("load test dict failed")

    print("summary_dict keys:", summary_dict.keys())

    # eval_steps = np.cumsum(summary_dict["step"][:,0])
    eval_steps = summary_dict["step"][:, 0]
    for key in keys:
        if "_test" in key:
            try:
                expt_dict[key] = {
                    "step": [0, 1],
                    "value": [test_dict[key], test_dict[key]],
                }
            except Exception:
                print("issue with loading :", key, expt_dict["dir"])
        else:
            try:
                expt_dict[key] = {"step": eval_steps, "value": summary_dict[key][:, 0]}
            except Exception:
                print("issue with loading :", key, expt_dict["dir"])

    return expt_dict
