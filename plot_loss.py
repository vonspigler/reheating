import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import decimal

COLD_LR, COLD_BS = '0.0256', '128'
SIMULATION_TYPE = 'fixed_bs'  # either 'fixed_bs' or 'fixed_lr'
OUTPUT_DIR = 'reheating_data/' + SIMULATION_TYPE + '_cold_lr=' + COLD_LR + '_bs=' + COLD_BS

def get_lrbs_from_file(filename):
    lr, bs = tuple(os.path.splitext(filename)[0].split('_')[-2:])
    lr = decimal.Decimal(lr.split('=')[-1])
    bs = decimal.Decimal(bs.split('=')[-1])
    return lr, bs

def pickle_everything(filename):
    losses = []

    with open(filename, 'rb') as dump:
        losses = []

        while True:
            try:
                losses.append(pickle.load(dump))
            except EOFError:
                break

    return losses

# -- Load the loss data -- #

cold_losses = []
all_reheated_losses = []

# NOTE: random order while loading...
print("Loading data...")
for file in os.listdir(OUTPUT_DIR):
    if fnmatch.fnmatch(file, 'cold_losses_*.p'):
        lr, bs = get_lrbs_from_file(file)
        losses = pickle_everything(OUTPUT_DIR + '/' + file)
        cold_losses = [lr, bs, losses]

    elif fnmatch.fnmatch(file, 'reheated_losses_*.p'):
        lr, bs = get_lrbs_from_file(file)
        losses = pickle_everything(OUTPUT_DIR + '/' + file)
        all_reheated_losses.append([lr, bs, losses])

if len(cold_losses) == 0:
    print("Cold losses not found!")
    quit()


# -- Plot the loss data -- #


_, (plt_left, plt_right) = plt.subplots(1, 2, sharey = True, figsize = (16.8, 10))
plt.tight_layout()

print("Plotting cold run...")
lr = cold_losses[0]
plt_left.plot([ t*lr for t, l in cold_losses[-1] ], [ l for t, l in cold_losses[-1] ], '-')
plt_left.set_yscale('log')
plt_left.set_xscale('log')
plt_left.set_ylabel('Loss')
plt_left.set_xlabel('$\lambda$t')

print("Plotting reheated losses...")
for reheated_losses in all_reheated_losses:
    lr, bs = reheated_losses[0:2]
    plt_right.plot(
        [ t*lr for t, l in reheated_losses[-1] ], [ l for t, l in reheated_losses[-1] ], '-',
        label = "T={}, lr={}, bs={}".format(lr*bs, lr, bs)
    )
plt_right.set_yscale('log')
plt_right.set_xscale('log')
plt_right.set_xlabel('$\lambda$t')
plt_right.legend()

#plt.show()
with PdfPages(OUTPUT_DIR + '/all_losses.pdf') as pdf:
    pdf.savefig()

plt.close()


# -- Plot the loss data (normalized) -- #


plt.figure(figsize = (16.8, 10))
plt.tight_layout()

print("Plotting reheated losses (normalized)...")
for reheated_losses in all_reheated_losses:
    lr, bs = reheated_losses[0:2]
    plt.plot(
        [ t*lr for t, l in reheated_losses[-1] ], [ float(l)/l0 for (t, l), (_, l0) in zip(reheated_losses[-1], all_reheated_losses[0][-1]) ], '-',
        label = "T={}, lr={}, bs={}".format(lr*bs, lr, bs)
    )
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\lambda$t')
plt.legend()

#plt.show()
with PdfPages(OUTPUT_DIR + '/all_losses_normalized.pdf') as pdf:
    pdf.savefig()

plt.close()
