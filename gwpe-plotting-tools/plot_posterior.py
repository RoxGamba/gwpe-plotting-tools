"""
Make corner plots of bilby/RIFT posteriors.
Example usage:
python plot_posterior.py -i /path/to/injrec_gr_ecc_abhf_dc.hdf5 /path/to/injrec_gr_ecc_abhf_rg.hdf5 \
	                     -p chirp_mass mass_ratio chi_eff -r '[27, 29.5]' '[0.25,0.45]' '[0.25, 0.55]' \
			             -p eccentricity mean_per_ano -r '[0.06, 0.14]' '[0, np.pi]' \
			             -l DC RG
This will make two corner plots, each comparing the results from the two hdf5 inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib import gridspec
import plot_utils as plu
import seaborn as sns
import argparse
import os
from pesummary.core.plots.figure import figure
try:
    import palettable
    palettableQ = True
except:
    palettableQ = False

mdc_to_bilby = {
            'm1': 'mass_1',             'm2': 'mass_2',
            'a1x': 'spin_1x',           'a1y': 'spin_1y',
            'a1z': 'spin_1z',           'a2x': 'spin_2x',
            'a2y': 'spin_2y',           'a2z': 'spin_2z',
            'mc': 'chirp_mass',         'eta': 'symmetric_mass_ratio',
            'ra': 'ra',                 'dec': 'dec',
            'tref': 'geocent_time',     'phiorb': 'phase',
            'incl': 'theta_jn',         'psi': 'psi',
            'dist': 'luminosity_distance',
            'Npts': 'npts',            'lnL': 'log_likelihood',
            'p': 'p',                  'ps': 'ps',
            'neff': 'neff',            'mtotal': 'total_mass',
            'total_mass': 'total_mass', 'mass_ratio': 'mass_ratio',
            'q': 'mass_ratio',         'chi_eff': 'chi_eff',
            'chi_p': 'chi_p',          'm1_source': 'mass_1_source',
            'm2_source': 'mass_2_source',       'mc_source': 'chirp_mass_source',
            'mtotal_source': 'total_mass_source',    'redshift': 'redshift',
            'eccentricity': 'eccentricity',          'meanPerAno': 'mean_per_ano',
            'indx': 'indx',
            'delta_taulm0': 'delta_taulm0', 'delta_omglm0': 'delta_omglm0',
            'iota': 'iota',
            'delta_Mbhf': 'delta_Mbhf', 'delta_abhf': 'delta_abhf'
        }

def read_truth(filename, kind='bilby-hdf5'):
    """
    Read truth dictionary from config for injections.
    """
    pardic = None
    if 'bilby' in kind:
        if 'ini' in filename:
            with open(filename, 'r') as f:
                for line in f:
                    if "injection-dict" not in line:
                        continue
                    pardic = eval(line[15:])
        elif 'json' in filename:
            import json
            with open(filename, 'r') as f:
                pardic = json.load(f)
    if (kind == 'rift' or pardic is None):
        injpars = np.genfromtxt(filename, dtype=float, names=True)
        pardic  = {mdc_to_bilby[key]: injpars[key] for key in injpars.dtype.names}
        if 'eccentricity' not in pardic.keys():
            pardic['eccentricity'] = 0.
        if 'mean_per_ano' not in pardic.keys():
            pardic['mean_per_ano'] = 0.
    if pardic is None:
        raise ValueError("Unrecognized input kind.")
    
    fillout_pardic(pardic)
    return pardic

def fillout_pardic(pardic):
    """
    Fill out a parameter dictionary assuming that the mass ratio, spins, theta_jn, 
    and one of M or Mc are already there. Surely incomplete.
    """
    keylist = pardic.keys()
    if 'chi_1' not in keylist and 'spin_1z' in keylist:
        pardic['chi_1'] = pardic['spin_1z']
    if 'chi_2' not in keylist and 'spin_2z' in keylist:
        pardic['chi_2'] = pardic['spin_2z']
    # Maybe add conversion from spin components to angles for precession...
    if 'chi_eff' not in keylist and 'chi_1' in keylist and 'chi_2' in keylist and 'mass_ratio' in keylist:
        pardic['chi_eff'] = (pardic['chi_1'] + pardic['mass_ratio']*pardic['chi_2'])/(1. + pardic['mass_ratio'])
    if 'chi_eff' not in keylist and 'a_1' in keylist and 'a_2' in keylist and 'mass_ratio' in keylist and 'tilt_1' in keylist and 'tilt_2' in keylist:
        pardic['chi_eff'] = (pardic['a_1']*np.cos(pardic['tilt_1']) + pardic['mass_ratio']*pardic['a_2']*np.cos(pardic['tilt_2']))/(1. + pardic['mass_ratio'])
    if 'symmetric_mass_ratio' not in keylist and 'mass_ratio' in keylist:
        pardic['symmetric_mass_ratio'] = pardic['mass_ratio']/(1. + pardic['mass_ratio'])**2
    if 'total_mass' not in keylist and 'chirp_mass' in keylist and 'symmetric_mass_ratio' in keylist:
        pardic['total_mass'] = pardic['chirp_mass']*pardic['symmetric_mass_ratio']**(-0.6)
    if 'chirp_mass' not in keylist and 'total_mass' in keylist and 'symmetric_mass_ratio' in keylist:
        pardic['chirp_mass'] = pardic['total_mass']*pardic['symmetric_mass_ratio']**(0.6)
    if 'mass_1' not in keylist and 'total_mass' in keylist and 'mass_ratio' in keylist:
        pardic['mass_1']  = pardic['total_mass']/(1. + pardic['mass_ratio'])
    if 'mass_2' not in keylist and 'total_mass' in keylist and 'mass_ratio' in keylist:
        pardic['mass_2']  = pardic['mass_ratio']*pardic['total_mass']/(1. + pardic['mass_ratio'])
    if 'cos_theta_jn' not in keylist and 'theta_jn' in keylist:
        pardic['cos_theta_jn'] = np.cos(pardic['theta_jn'])
    if 'chi_p' not in keylist:
        if 'tilt_1' in keylist:
            pardic['chi_p'] = plu.compute_chi_prec(pardic['mass_1'], pardic['mass_2'], pardic['a_1'], pardic['a_2'], pardic['tilt_1'], pardic['tilt_2'])
        elif 'spin_1x' in keylist and 'spin_1y' in keylist and 'spin_2x' in keylist and 'spin_2y' in keylist:
            pardic['chi_p'] = plu.compute_chi_prec_from_xyz(pardic['mass_ratio'], pardic['spin_1x'], pardic['spin_1y'], pardic['spin_2x'], pardic['spin_2y'])
    for par, other in zip(['alpha', 'tau'], ['tau', 'alpha']):
        if f'delta_{par}lm0' in keylist:
            if pardic[f'delta_{par}lm0'] != 0. and pardic[f'delta_{other}lm0'] == 0.:
                pardic[f'delta_{other}lm0'] = -1. + 1/(1 + pardic[f'delta_{par}lm0'])

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',  '--input', type=str,   help='Path to hdf5 (bilby)/extrinsic_posterior_samples.dat (rift) data file(s).', required=True, nargs='*')
    parser.add_argument('-t',  '--truth', type=str,   help='Truth file(s) for injections; if not provided, looking for config with similar name to posterior file.', default=None, nargs='*')
    parser.add_argument('-l',  '--label', type=str,   help='Labels for legends.', nargs='*', default=None)
    parser.add_argument('-p',  '--par',   type=str,   help='For each corner plot, add an instance specifying what parameters it should plot. E.g.: plot_posterior.py -i res.hdf5 -p chirp_mass chi_eff -p eccentricity mean_per_ano', default=None, action='append', nargs='*')
    parser.add_argument('-r',  '--range', type=str,   help='Parameter ranges for every par in every corner; usage similar to par.', action='append', nargs='*')
    parser.add_argument('-k',  '--ticks', type=str,   help='Ticks for every par in every corner; usage similar to par.', action='append', nargs='*', default=None)
    parser.add_argument('-s',  '--save',              help='Save plot(s)?', action='store_true')
    parser.add_argument('-tr', '--triangle',          help='Make triangle plots with pesummary (when possible).', action='store_true')
    parser.add_argument('-n',  '--name',  type=str,   help='Name for figures', default=None)
    parser.add_argument('--size',         type=float, help='Size of plot', default=3)

    args = parser.parse_args()

    Np = len(args.input)

    if args.truth is None:
        args.truth = ['' for jj in range(Np)]

    if palettableQ:
        cmap = palettable.cartocolors.diverging.Tropic_7.mpl_colormap
    else:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("beyond", ["darkturquoise", "aliceblue", "coral"])
    # base_cols  = sns.color_palette("colorblind", n_colors=Np)
    base_cols = cmap(np.array(range(Np))/(max(Np - 1, 1)))
    # base_cols = [plu.p_green, plu.p_pink, plu.p_purple]
    base_cols = [plu.p_green, plu.p_pink, plu.p_green, plu.p_pink]
    line_cols  = []
    truth_cols = []
    for col in base_cols:
        line_cols.append('#' + ''.join([hex(int(c*255))[2:].zfill(2) for c in col]))
        truth_cols.append('#' + ''.join([hex(int(0.5*c*255))[2:].zfill(2) for c in col]))
    if Np == 1 or True:
        truth_cols = ['k']*Np

    figs    = []
    handles = []

    for ji, (infile, truthfile, color, truthcolor) in enumerate(zip(args.input, args.truth, line_cols, truth_cols)):

        if '.hdf5' in infile:
            inkind = 'bilby-hdf5'
        elif '.json' in infile:
            inkind = 'bilby-json'
        elif '.txt' in infile:
            inkind = 'txt'
        else:
            inkind = 'rift'
        post = plu.Posterior(infile, kind=inkind)
        if not truthfile:
            truthfile = infile.replace(infile.split('/')[-1].split('_')[0], 'config').replace('.hdf5', '.ini')
            print(f"Looking for truth file: {truthfile}.")
        if not os.path.exists(truthfile):
            print(f"Couldn't find truth file {truthfile}.")
            truths = None
        else:
            truths = read_truth(truthfile, kind=inkind)
        
        for jp, parlist in enumerate(args.par):
            if args.range is not None:
                lims = [eval(interval) for interval in args.range[jp]]
            else:
                lims = [[np.min(post.__getattribute__(par)), np.max(post.__getattribute__(par))] for par in parlist]
            if truths is not None:
                true_vals = [truths[par] for par in parlist]
            else:
                true_vals = None
            
            if len(parlist) == 2 and args.triangle:
                if jp >= len(figs):
                    fig, ax1, temp, ax2, ax3 = plu._triangle_axes(figsize=(args.size*2, args.size*2))
                    temp.remove()
                    ax1.set_yticks([])
                    ax1.grid(visible=False)
                    ax3.set_xticks([])
                    ax3.set_yticks([])
                    ax3.grid(visible=False)
                    figs.append(fig)
                post.make_triangle_plot(parlist, color,  
                                        args.label[ji] if args.label is not None else None, 
                                        fig=figs[jp], axes=figs[jp].get_axes(), 
                                        fill=True,
                                        plot_density=True,#(ji==0), 
                                        truth=[truths[par] for par in parlist] if truths is not None else None,
                                        N=10000, 
                                        percentiles=[5, 95],
                                        rangex=lims[0], rangey=lims[1],
                                        grid=True,
                                        legend_kwargs={'loc': 'upper right', 'frameon': True, 
                                                    #    'bbox_to_anchor': (1.025,0),
                                                        'framealpha': 0.75, 'facecolor': 'w', 'edgecolor': 'w', 'fontsize': 12}
                                        )
            else:
                if jp >= len(figs):
                    # figs.append(plt.figure(layout='tight', figsize=(args.size*(2 + (len(parlist) - 2)), args.size*(2 + (len(parlist) - 2))), gca=False))
                    figs.append(figure(figsize=(args.size*(2 + (len(parlist) - 2)), args.size*(2 + (len(parlist) - 2))), gca=False))
                    gs = gridspec.GridSpec(len(parlist), len(parlist), width_ratios=[1]*len(parlist), height_ratios=[1]*len(parlist),
                                           wspace=0.0, hspace=0.0)
                    axes = [figs[jp].add_subplot(gs[jjj]) for jjj in range(len(parlist)**2)]
                    for jj in range(len(parlist)):
                        axes[jj*len(parlist)].minorticks_on()
                        if jj >= 0:
                            for ii in range(jj+1):
                                axes[jj*len(parlist) + ii].tick_params(direction='in', which='both', top=False, right=False)
                    axes[0].xaxis.set_ticklabels([])
                ylims_now = [this_ax.get_ylim()[-1] for this_ax in [axes[ii*(len(parlist) + 1)] for ii in range(len(parlist))]]
                figs[jp], _ = post.make_corner_plot(parlist, lims, color,
                                      truths=true_vals, truth_color=truthcolor, fig=figs[jp])
                if args.ticks is not None:
                    for jpar, par in enumerate(parlist):
                        if args.ticks[jp][jpar] is not None:
                            tickvals = eval(args.ticks[jp][jpar])
                            # Diagonal
                            axdiag = figs[jp].get_axes()[jpar*len(parlist) + jpar]
                            if par == 'mean_per_ano' and False:
                                axdiag.set_xticks(tickvals, labels=[r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
                            else:
                                axdiag.set_xticks(tickvals)
                            # Off-diagonal x
                            for jrow in range(jpar + 1, len(parlist)):
                                axoffx = figs[jp].get_axes()[jrow*len(parlist) + jpar]
                                if par == 'mean_per_ano' and False:
                                    axoffx.set_xticks(tickvals, labels=[r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
                                else:
                                    axoffx.set_xticks(tickvals)
                            # Off-diagonal y
                            for jcol in range(jpar):
                                axoffy = figs[jp].get_axes()[jpar*len(parlist) + jcol]
                                if par == 'mean_per_ano' and False:
                                    axoffy.set_yticks(tickvals, labels=[r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
                                else:
                                    axoffy.set_yticks(tickvals)
                for ii in range(len(parlist)):
                    axdiag = figs[jp].get_axes()[ii*len(parlist) + ii]
                    axdiag.set_ylim(0, max(ylims_now[ii], axdiag.get_ylim()[-1]))
    
    if args.label is not None:
        for jf, fig in enumerate(figs):
            if not (args.triangle and len(args.par[jf]) == 2):
                fig.legend(handles=[matplotlib.lines.Line2D([], [], color=color, label=label) for color, label in zip(line_cols, args.label)], frameon=False, fontsize=15, loc='upper right')
    
    for fig, parlist in zip(figs, args.par):
            for any_ax in fig.get_axes():
                any_ax.tick_params(labelsize=9)
    
    if args.save:
        if args.name is None:
            namestr = ''
            for file in args.input:
                namestr = namestr + file.split('/')[-1].split('.')[-2] + '_'
        else:
            namestr = args.name
        for fig, parlist in zip(figs, args.par):
            for artist in fig.get_children():
                try:
                    artist.set_rasterized(True)
                except AttributeError:
                    pass
            fig.savefig('/Users/apple/Documents/EOB/teob-parametrized/py/figs/pe/' + namestr + '_'.join((parname for parname in parlist)) + '.png', bbox_inches='tight')
            fig.savefig('/Users/apple/Documents/EOB/teob-parametrized/py/figs/pe/' + namestr + '_'.join((parname for parname in parlist)) + '.pdf', bbox_inches='tight', dpi=300)
    else:
        plt.show()