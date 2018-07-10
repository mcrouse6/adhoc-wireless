import torch
import numpy as np
import matplotlib.pyplot as plt


def simulate_args_from_namespace(n, positional=[]):
    """ check an argparse namespace against a module's get_args method.
    Ideally, there would be something built in to argparse, but no such luck.
    This tries to reconstruct the arg list that argparse.parse_args would expect
    """
    arg_list = [[k, v] for k, v in sorted(vars(n).items())]
    argparse_formatted_list = []
    for l in arg_list:
        ####  deal with flag arguments (store true/false)
        if l[1] == True:
            argparse_formatted_list.append("{}".format(l[0]))
        elif l[1] == False or l[1] is None:
            pass  # dont add this arg
        # add positional argments
        elif l[0] in positional:
            argparse_formatted_list.append(str(l[0]))
        # add the named arguments
        else:
            argparse_formatted_list.append("{}".format(l[0]))
            argparse_formatted_list.append(str(l[1]))
    return argparse_formatted_list

def cdf(data):

    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)

    # Plot the cdf
    plt.plot(bin_edges[0:-1], cdf,linestyle='--', marker="o", color='b')
    plt.ylim((0,1))
    plt.ylabel("CDF")
    plt.grid(True)

    plt.show()