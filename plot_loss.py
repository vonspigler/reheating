import os
import fnmatch
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.backends.backend_pdf import PdfPages
import decimal

def get_lrbs_from_file(filename):
    """Extract LR and BS from a file's name.

    The filename has the format ''.../..._lr={LR}_bs={BS}.{extension}.'
    """

    lr, bs = tuple(os.path.splitext(filename)[0].split('_')[-2:])
    lr = decimal.Decimal(lr.split('=')[-1])
    bs = decimal.Decimal(bs.split('=')[-1])
    return lr, bs

def pickle_everything(filename):
    """This function opens the file and loads all the pickles inside, until it
    is empty (EOF).

    Pickled objects are returned in a list.
    """

    losses = []

    with open(filename, 'rb') as dump:
        losses = []

        while True:
            try:
                losses.append(pickle.load(dump))
            except EOFError:
                break

    return losses


# --  Load the loss data  ---------------------------------------------------- #


#COLD_LR, COLD_BS, SIMULATION_TYPE = '0.0256', '128', 'fixed_bs'
COLD_LR, COLD_BS, SIMULATION_TYPE = '0.03', '150', 'fixed_lr'
OUTPUT_DIR = 'reheating_data/' + SIMULATION_TYPE + '_cold_lr=' + COLD_LR + '_bs=' + COLD_BS

cold_losses = []
all_reheated_losses = []

print("Loading data...")
for filename in os.listdir(OUTPUT_DIR):
    if fnmatch.fnmatch(filename, 'cold_losses_*.p'):
        lr, bs = get_lrbs_from_file(filename)
        losses = pickle_everything(OUTPUT_DIR + '/' + filename)
        cold_losses = [lr, bs, losses]

    elif fnmatch.fnmatch(filename, 'reheated_losses_*.p'):
        lr, bs = get_lrbs_from_file(filename)
        losses = pickle_everything(OUTPUT_DIR + '/' + filename)
        all_reheated_losses.append([lr, bs, losses])

if len(cold_losses) == 0:
    print("Cold losses not found!")
    quit()

all_reheated_losses = sorted(all_reheated_losses, key = lambda el: el[0]/el[1])


# --  Plot the loss data  ---------------------------------------------------- #


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
    if len(reheated_losses[-1]) == 0: continue  # the simulation is still running

    lr, bs = reheated_losses[0:2]
    if not reheated_losses[-1][-1][1] > 0:
        print("Skipping lr={}, bs={} -- diverged".format(lr, bs))
        continue

    plt_right.plot(
        [ t*lr for t, l in reheated_losses[-1] ], [ l for t, l in reheated_losses[-1] ], '-',
        label = "T={:.2}, lr={}, bs={}".format(lr/bs, lr, bs)
    )
plt_right.set_yscale('log')
plt_right.set_xscale('log')
plt_right.set_xlabel('$\lambda$t')
plt_right.legend()

#plt.show()
with PdfPages(OUTPUT_DIR + '/all_losses.pdf') as pdf:
    pdf.savefig()

plt.close()


# --  Plot the loss data (normalized)  --------------------------------------- #


plt.figure(figsize = (16.8, 10))
plt.tight_layout()

print("Plotting reheated losses (normalized)...")
for reheated_losses in all_reheated_losses:
    if len(reheated_losses[-1]) == 0: continue  # the simulation is still running

    lr, bs = reheated_losses[0:2]
    if not reheated_losses[-1][-1][1] > 0:
        print("Skipping lr={}, bs={} -- diverged".format(lr, bs))
        continue

    plt.plot(
        [ t*lr for t, l in reheated_losses[-1] ], [ float(l)/l0 for (t, l), (_, l0) in zip(reheated_losses[-1], all_reheated_losses[0][-1]) ], '-',
        label = "T={:.2}, lr={}, bs={}".format(lr/bs, lr, bs)
    )
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\lambda$t')
plt.legend()

#plt.show()
with PdfPages(OUTPUT_DIR + '/all_losses_normalized.pdf') as pdf:
    pdf.savefig()

plt.close()
