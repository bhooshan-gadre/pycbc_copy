import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_cmd_line():

    desc = """
Make corner plot of paramspace.
"""
    parser = argparse.ArgumentParser(description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trigger-file', '-t', type=str, required=True,
            help='dat file containing params.')
    parser.add_argument('--out-file', '-o', type=str, required=True,
            help='filename to plot to. Should have .png or .pdf extension.')
    parser.add_argument('--x-axis', '-x', type=str,help='should exist in hdf file',
                    required=True)
    parser.add_argument('--y-axis', '-y', type=str, help='should exist in hdf file',
                    required=True)
    parser.add_argument("--log-scale", action="store_true",
                  help="Makes axes scale log-log", default=False )
    args = parser.parse_args()
    return args

args = parse_cmd_line()

f = h5py.File(args.trigger_file, 'r')
x_val = f[args.x_axis][()]
y_val = f[args.y_axis][()]
f.close()

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(x_val, y_val, 'r.')
ax.set_xlim(x_val.min(), x_val.max())
ax.set_ylim(y_val.min(), y_val.max())
if args.log_scale:
    ax.set_xscale('log')
    ax.set_yscale('log')
# plt.legend(loc='best', frameon=0)
plt.xlabel(args.x_axis, fontsize=14)
plt.ylabel(args.y_axis, fontsize=14)
plt.savefig(args.out_file, bbox_inches='tight')
plt.close()
