import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.ticker as tck

import os
import argparse

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.seba.interface import SeBa
from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file, read_set_from_file

from venice import Venice, write_set_to_amuse_file

plt.rcParams.update({'font.size': 12})


def break_string (string, max_words=3):

    split_string = string.split(' ')
    res = ''
    counter = 0
    for s in split_string:
        res += s
        if counter == max_words-1:
            res += '\n'
        else:
            res += ' '
        counter += 1; counter %= max_words

    return res[:-1]


def setup_gravity_stellar (timescale):

    converter = nbody_system.nbody_to_si(timescale, 1.|units.MSun)
    gravity = ph4(converter)
    gravity.parameters.force_sync = True

    stellar = SeBa()

    system = Venice()
    system.record_runtime = True

    system.add_code(gravity)
    system.add_code(stellar)

    system.add_channel(1, 0, from_attributes=['mass'], to_attributes=['mass'])

    system.timescale_matrix[0,1] = timescale

    return system, gravity, stellar


def setup_gravity_stellar_reverse (timescale):

    converter = nbody_system.nbody_to_si(timescale, 1.|units.MSun)
    gravity = ph4(converter)
    gravity.parameters.force_sync = True

    stellar = SeBa()

    system = Venice()
    system.record_runtime = True

    system.add_code(stellar)
    system.add_code(gravity)

    system.add_channel(0, 1, from_attributes=['mass'], to_attributes=['mass'])

    system.timescale_matrix[0,1] = timescale

    return system, gravity, stellar


def get_stellar_timescale (stellar):

    stellar.channel.copy()
    return stellar.local_particles.time_step.min()


def setup_gravity_stellar_adaptive (eta):

    converter = nbody_system.nbody_to_si(1.|units.Myr, 1.|units.MSun)
    gravity = ph4(converter)
    gravity.parameters.force_sync = True

    stellar = SeBa()

    system = Venice()
    system.record_runtime = True

    system.add_code(gravity)
    system.add_code(stellar)

    system.add_channel(1, 0, from_attributes=['mass'], to_attributes=['mass'])

    system.timescale_matrix[0,1] = 1.|units.kyr

    system.update_timescale[0][1] = lambda gravity, stellar, dt: \
        get_stellar_timescale(stellar) * eta
    # The line below also works, but is not optimal, as it opens a new MPI channel every call.
    # By making a channel we open one MPI channel and keep it open
    #    stellar.particles.time_step.min() * eta

    return system, gravity, stellar


def run_convergence_gravostellar (stars, timescales, end_time, orders=[1,2], output_path=None):

    for i in range(len(orders)):

        timer_framework = np.zeros(len(timescales))
        timer_codes = np.zeros((len(timescales), 2))

        for j in range(len(timescales)):

            print (timescales[j].value_in(units.kyr), flush=True)

            system, gravity, stellar = setup_gravity_stellar(timescales[j])
            gravity.particles.add_particles(stars.copy())
            stellar.particles.add_particles(stars.copy())

            system.rest_order = orders[i]

            if output_path is not None:
                system.io_scheme = 2
                system.filepath = output_path + '/gravo_stellar_dt_{a}_{b}/'.format(a=timescales[j].value_in(units.kyr), b=orders[i])
                system.save_data[0] = lambda code, filename: write_set_to_amuse_file(
                    code, filename, 'gravity')

                if not os.path.exists(system.filepath):
                    os.mkdir(system.filepath)

            system.evolve_model(end_time)

            timer_framework[j] = system.runtime_framework
            timer_codes[j] = system.runtime_codes

            system.stop()

        np.savetxt('data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            np.array([timescales.value_in(units.kyr), timer_framework,
                timer_codes[:,0], timer_codes[:,1]]).T)


def run_convergence_gravostellar_reversed (stars, timescales, end_time, orders=[1,2]):

    for i in range(len(orders)):

        timer_framework = np.zeros((2, len(timescales)))
        timer_codes = np.zeros((2, len(timescales), 2))

        for j in range(len(timescales)):

            print (timescales[j].value_in(units.kyr), flush=True)


            system, gravity, stellar = setup_gravity_stellar(timescales[j])
            gravity.particles.add_particles(stars.copy())
            stellar.particles.add_particles(stars.copy())

            system.rest_order = orders[i]

            system.evolve_model(end_time)

            timer_framework[0,j] = system.runtime_framework
            timer_codes[0,j] = system.runtime_codes

            write_set_to_file(gravity.particles, 
                'data/gravostellar_reversed_final_dt_{a}_{b}.hdf5'.format(a=timescales[j].value_in(units.kyr), b=orders[i]), 
                'hdf5', overwrite_file=True)

            system.stop()


            system_r, gravity_r, stellar_r = setup_gravity_stellar_reverse(timescales[j])
            gravity_r.particles.add_particles(stars.copy())
            stellar_r.particles.add_particles(stars.copy())

            system_r.rest_order = orders[i]

            system_r.evolve_model(end_time)

            timer_framework[1,j] = system_r.runtime_framework
            timer_codes[1,j] = system_r.runtime_codes

            write_set_to_file(gravity_r.particles, 
                'data/gravostellar_reversed_final_r_dt_{a}_{b}.hdf5'.format(a=timescales[j].value_in(units.kyr), b=orders[i]), 
                'hdf5', overwrite_file=True)

            system_r.stop()

        np.savetxt('data/timer_gravostellar_reversed_order_{a}.txt'.format(
                a=orders[i]),
            np.array([timescales.value_in(units.kyr), 
                timer_framework[0], timer_codes[0,:,0], timer_codes[0,:,1],
                timer_framework[1], timer_codes[1,:,0], timer_codes[1,:,1]]).T)


def run_convergence_gravostellar_adaptive (stars, eta, end_time, orders=[1,2]):

    for i in range(len(orders)):

        timer_framework = np.zeros(len(eta))
        timer_codes = np.zeros((len(eta), 2))

        for j in range(len(eta)):

            print (eta[j], flush=True)

            system, gravity, stellar = setup_gravity_stellar_adaptive(eta[j])
            gravity.particles.add_particles(stars.copy())
            stellar.particles.add_particles(stars.copy())

            stellar.local_particles = stars.copy()
            stellar.channel = stellar.particles.new_channel_to(
                stellar.local_particles, attributes=['time_step'])

            system.rest_order = orders[i]
            system.record_timescales = True

            system.evolve_model(end_time)

            timer_framework[j] = system.runtime_framework
            timer_codes[j] = system.runtime_codes

            write_set_to_file(gravity.particles, 
                'data/gravostellar_adaptive_final_eta_{a}_{b}.hdf5'.format(a=eta[j], b=orders[i]), 
                'hdf5', overwrite_file=True)

            np.savetxt('data/timescales_eta_{a}_{b}.txt'.format(a=eta[j], b=orders[i]), 
                system.timescales_codes[0].value_in(units.kyr))

            system.stop()

        np.savetxt('data/timer_gravostellar_adaptive_{a}.txt'.format(a=orders[i]),
            np.array([eta, timer_framework, timer_codes[:,0], timer_codes[:,1]]).T)


def plot_convergence_gravostellar (output_path, orders=[1,2]):

    fig = plt.figure(figsize=(12.8, 9.6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale [kyr]')
    ax1.set_ylabel('Median position error [pc]')

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    ax1.plot(0., 0., c='C0', label='1$^\\mathrm{st}$ order')
    ax1.plot(0., 0., c='C1', label='2$^\\mathrm{nd}$ order')

    ax1.set_xlim(3e-2, 3e2)
    ax1.set_ylim(1e-10, 1e-5)

    ax1.set_xticks([1e-1, 1e0, 1e1, 1e2])
    ax1.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Run-time [s]')
    ax2.set_ylabel('Median position error [pc]')

    ax2.set_xlim(3e-1, 1e3)
    ax2.set_ylim(1e-10, 1e-5)

    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    ax2.set_xticks([1e0, 1e1, 1e2, 1e3])
    ax2.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])


    ax3.set_xlabel('Coupling timescale [kyr]')
    ax3.set_ylabel('Run-time [s]')

    ax3.set_xscale('log')
    ax3.set_yscale('log')

    ax3.set_xlim(3e-2, 3e2)
    ax3.set_ylim(3e-2, 1e3)

    ax3.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax3.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax3.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax3.set_yticks([1e-1, 1e0, 1e1, 1e2])


    ax4.set_yscale('log')

    ax4.set_xlabel('t [Myr]')
    ax4.set_ylabel('Median position error [pc]')

    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()

    ax4.set_xlim(0., 6.)
    ax4.set_ylim(1e-15, 1e-5)

    ax4.xaxis.set_minor_locator(tck.MultipleLocator(0.25))


    ls = ['--', '-.']
    ls2 = ['-', '--', '-.', ':']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        dR_med = np.zeros(len(timescales)-1) | units.pc

        filepath = output_path + '/gravo_stellar_dt_{a}_{b}/'.format(a=timescales[0], b=orders[i])
        files1 = os.listdir(filepath)
        files1 = list(filter(lambda f: f.split('_')[0] == 'plt', files1))
        files1.sort(key=lambda f: f.split('.')[-2][-6:])

        stars1 = [ read_set_from_file(filepath+f, 'hdf5') for f in files1 ]

        for j in range(1,len(timescales)):

            print (i, j, flush=True)

            filepath = output_path + '/gravo_stellar_dt_{a}_{b}/'.format(a=timescales[j], b=orders[i])
            files2 = os.listdir(output_path + '/gravo_stellar_dt_{a}_{b}/'.format(a=timescales[j], b=orders[i]))
            files2 = list(filter(lambda f: f.split('_')[0] == 'plt', files2))
            files2.sort(key=lambda f: f.split('.')[-2][-6:])

            skip = len(files1)//len(files2)

            stars2 = [ read_set_from_file(filepath+f, 'hdf5') for f in files2 ]

            dR_med[j-1] = (stars1[-1].position - stars2[-1].position).lengths().median()

            t = np.zeros(len(stars2)) | units.Myr
            dR = np.zeros(len(stars2)) | units.pc

            for k in range(len(stars2)):
                t[k] = stars2[k].get_timestamp()
                dR[k] = (stars1[(k+1)*skip-1].position - stars2[k].position).lengths().median()

            ax4.plot(t.value_in(units.Myr), dR.value_in(units.pc), c='C'+str(i), ls=(0, (j, j)))

            files1 = files2
            stars1 = stars2

        timescales_mid = (timescales[1:]*timescales[:-1])**0.5
        ax1.errorbar(timescales_mid, dR_med.value_in(units.pc),
            xerr=(timescales_mid-timescales[:-1], timescales[1:]-timescales_mid),
            capsize=3., fmt='C{a},:'.format(a=i))

        ax1.plot(timescales_mid, 
            dR_med[len(timescales_mid)//2].value_in(units.pc) * (timescales_mid/timescales_mid[len(timescales_mid)//2])**orders[i],
            c='k', ls='--')
        if i == 0:
            ax1.text(2e-1, 6e-9, 'dR~$\\tau^1$', rotation=30)
        if i == 1:
            ax1.text(6e-1, 5e-10, 'dR~$\\tau^2$', rotation=45)


        timer = timer_framework + timer_gravity + timer_stellar
        timer_mid = (timer[1:]*timer[:-1])**0.5

        ax2.errorbar(timer_mid, dR_med.value_in(units.pc),
            xerr=(timer_mid-timer[1:], np.abs(timer[:-1]-timer_mid)),
            capsize=3., fmt='C{a},:'.format(a=i))

        if i == 1:
            ax2.text(timer[0], dR_med[0].value_in(units.pc)/1.2, 
                '$\\tau$={b}-{a} kyr'.format(a=timescales[0], b=timescales[1]), ha='right', va='top')
            ax4.text(t[-1].value_in(units.Myr), 1e-10, 
                '$\\tau$={b}-{a} kyr'.format(a=timescales[0], b=timescales[1]), ha='right', va='top')
        if i == 0:
            ax2.text(timer[-1], dR_med[-1].value_in(units.pc)*1.2, 
                '$\\tau$={a}-{b} kyr'.format(a=str(timescales[-1])[:-2], b=str(timescales[-2])[:-2]))
            ax4.text(2., 3e-6, 
                '$\\tau$={a}-{b} kyr'.format(a=str(timescales[-1])[:-2], b=str(timescales[-2])[:-2]))


        ax3.plot(timescales, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax3.plot(timescales, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax3.plot(timescales, timer_stellar, 
            linestyle=':', c='C'+str(i))

    ax1.legend(frameon=False)
    ax3.legend(frameon=False)

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('figures/gravo_stellar_constant_convergence.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_constant_convergence.pdf', bbox_inches='tight')


def plot_convergence_gravostellar_reversed (orders=[1, 2]):

    fig = plt.figure(figsize=(6.4, 9.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale [kyr]')
    ax1.set_ylabel('Median position error [pc]')

    ax1.plot(0., 0., c='C0', label='1$^\\mathrm{st}$ order')
    ax1.plot(0., 0., c='C1', label='2$^\\mathrm{nd}$ order')

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    ax1.set_xlim(3e-2, 3e2)
    ax1.set_ylim(1e-10, 3e-5)


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Coupling timescale [kyr]')
    ax2.set_ylabel('Run-time [s]')

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax2.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax2.set_xlim(3e-2, 3e2)
    ax2.set_ylim(1e-2, 3e3)


    ls = ['--', '-.']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar, timer_framework_r, \
            timer_stellar_r, timer_gravity_r = np.loadtxt(
            'data/timer_gravostellar_reversed_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        dR_med = np.zeros(len(timescales)) | units.pc
        timescales_mid = (timescales[1:]*timescales[:-1])**0.5

        for j in range(len(timescales)):

            stars = read_set_from_file('data/gravostellar_reversed_final_dt_{a}_{b}.hdf5'.format(a=timescales[j], b=orders[i]),
                'hdf5')
            stars_r = read_set_from_file('data/gravostellar_reversed_final_r_dt_{a}_{b}.hdf5'.format(a=timescales[j], b=orders[i]),
                'hdf5')

            dR_med[j] = (stars.position - stars_r.position).lengths().median()

        ax1.plot(timescales, dR_med.value_in(units.pc), 'C{a}o:'.format(a=i))

        ax1.plot(timescales, dR_med[len(dR_med)//2].value_in(units.pc) * (timescales/timescales[len(dR_med)//2])**orders[i],
            c='k', ls='--')
        if i == 0:
            ax1.text(1e-1, 1e-8, 'dR~$\\tau^1$', rotation=30)
        elif i == 1:
            ax1.text(3e-1, 2e-10, 'dR~$\\tau^2$', rotation=45)

        ax2.plot(timescales, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax2.plot(timescales, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax2.plot(timescales, timer_stellar, 
            linestyle=':', c='C'+str(i))

        ax2.plot(timescales, timer_framework_r, 
            linestyle='-', c='C'+str(i))
        ax2.plot(timescales, timer_gravity_r, 
            linestyle='--', c='C'+str(i))
        ax2.plot(timescales, timer_stellar_r, 
            linestyle=':', c='C'+str(i))

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    fig.subplots_adjust(hspace=0)

    fig.savefig('figures/gravo_stellar_reverse_convergence.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_reverse_convergence.pdf', bbox_inches='tight')


def plot_convergence_gravostellar_adaptive (orders=[1, 2]):

    fig = plt.figure(figsize=(6.4, 9.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Timescale parameter $\\eta$')
    ax1.set_ylabel('Median position error [pc]')

    ax1.plot(0., 0., c='C0', label='1$^\\mathrm{st}$ order')
    ax1.plot(0., 0., c='C1', label='2$^\\mathrm{nd}$ order')

    ax1.set_xlim(3e-3, 3e1)
    ax1.set_ylim(3e-6, 1e-2)

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Timescale parameter $\\eta$')
    ax2.set_ylabel('Run-time [s]')

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax2.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax2.set_xlim(3e-3, 3e1)
    ax2.set_ylim(3e-2, 1e3)


    ls = ['--', '-.']
    for i in range(len(orders)):

        eta, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_adaptive_{a}.txt'.format(a=orders[i]),
            unpack=True)

        dR_med = np.zeros(len(eta)-1) | units.pc
        eta_mid = (eta[1:]*eta[:-1])**0.5

        stars1 = read_set_from_file('data/gravostellar_adaptive_final_eta_{a}_{b}.hdf5'.format(a=eta[0], b=orders[i]),
            'hdf5')

        for j in range(1,len(eta)):

            stars2 = read_set_from_file('data/gravostellar_adaptive_final_eta_{a}_{b}.hdf5'.format(a=eta[j], b=orders[i]),
                'hdf5')

            dR_med[j-1] = (stars1.position - stars2.position).lengths().median()

            stars1 = stars2

        ax1.errorbar(eta_mid, dR_med.value_in(units.pc),
            xerr=(eta_mid-eta[:-1], eta[1:]-eta_mid),
            capsize=3., fmt='C{a},:'.format(a=i))

        ax1.plot(eta_mid, dR_med[0].value_in(units.pc) * (eta_mid/eta_mid[0])**orders[i],
            c='k', ls='--')

        if i == 0:
            ax1.text(1e0, 3e-4, 'dR~$\\eta^1$', rotation=45)
        if i == 1:
            ax1.text(3e-1, 3e-3, 'dR~$\\eta^2$', rotation=60)

        ax2.plot(eta, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax2.plot(eta, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax2.plot(eta, timer_stellar, 
            linestyle=':', c='C'+str(i))

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    fig.subplots_adjust(hspace=0)

    fig.savefig('figures/gravo_stellar_adaptive_convergence.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_adaptive_convergence.pdf', bbox_inches='tight')


def plot_timescale_evolution_gravostellar_adaptive ():

    eta, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
        'data/timer_gravostellar_adaptive_1.txt', unpack=True)

    stellar = SeBa()
    stellar.particles.add_particles(Particles(1, mass=50.|units.MSun))

    stellar_type = [stellar.particles[0].stellar_type.value_in(units.stellar_type)]
    time = [0.]

    t = 0.|units.Myr
    eta_min = np.min(eta)

    timescales = np.loadtxt('data/timescales_eta_{a}_1.txt'.format(a=eta_min))
    end_time = np.sum(timescales)|units.kyr
    while t < end_time:
        dt = eta_min * stellar.particles[0].time_step
        if t+dt > end_time:
            dt = end_time - t
        stellar.evolve_model(t+dt)
        t += dt
        stellar_type.append(
            stellar.particles[0].stellar_type.value_in(units.stellar_type))
        time.append(t.value_in(units.Myr))

    fig = plt.figure(figsize=(12.8, 7.2))
    grd = gs.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[1., 0.6])
    ax1 = fig.add_subplot(grd[0])
    ax2 = fig.add_subplot(grd[1])

    ax1.set_yscale('log')

    ax1.set_xlabel('Time [Myr]')
    ax1.set_ylabel('Coupling timescale [yr]')

    ax1.set_xlim(0., 6.)
    ax1.set_ylim(1e-4, 1e6)

    ax1.tick_params(axis='x', bottom=False, labelbottom=False, 
        top=True, labeltop=True, which='both')
    ax1.xaxis.set_label_position('top')

    ax1.xaxis.set_minor_locator(tck.MultipleLocator(0.2))

    axins1 = ax1.inset_axes([0.5, 1e-3, 3.2, 1e2], transform=ax1.transData)
    axins1.set_xticks([4.2, 4.5, 4.8])
    axins1.set_yticklabels([])
    axins1.xaxis.set_minor_locator(tck.MultipleLocator(0.1))
    axins1.set_xlim(4.2, 4.8)
    axins1.set_ylim(1e2, 1e5)
    axins1.set_yscale('log')

    counter = 0
    for i in range(len(eta)):
        timescales = np.loadtxt('data/timescales_eta_{a}_1.txt'.format(a=eta[i]))
        ax1.plot(np.cumsum(timescales)/1e3, timescales*1e3, 
            ds='steps-pre', c='C'+str(len(eta)-counter-1), label='$\\eta$='+str(eta[i]))
        if eta[i] >= 1.:
            axins1.plot(np.cumsum(timescales)/1e3, timescales*1e3, 
                ds='steps-pre', c='C'+str(len(eta)-counter-1))
        counter += 1

    ax1.legend(loc='lower right', frameon=False, fontsize=10)

    ax1.indicate_inset_zoom(axins1, edgecolor='k')


    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Stellar type')

    stellar_types = np.unique(stellar_type)
    labels = [ break_string(str(stype|units.stellar_type)) for stype in stellar_types ]

    mapped_stellar_type = np.zeros(len(stellar_type))
    for i in range(len(stellar_types)):
        mapped_stellar_type[ stellar_type == stellar_types[i] ] = i

    ax2.plot(time, mapped_stellar_type, ds='steps-pre', c='k')

    ax2.set_yticks(np.arange(len(stellar_types)))
    ax2.tick_params(axis='y', pad=-10)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.set_yticklabels(labels, ha='right', va='bottom', fontsize=10)

    ax2.set_xlim(0., 6.)
    ax2.set_ylim(-0.5, len(stellar_types)-0.5)

    ax2.xaxis.set_minor_locator(tck.MultipleLocator(0.2))

    axins2 = ax2.inset_axes([0.5, -0.25, 3.2, len(stellar_types)-0.5], transform=ax2.transData)
    axins2.set_xticks([4.2, 4.5, 4.8])
    axins2.set_xticklabels([])
    axins2.xaxis.set_minor_locator(tck.MultipleLocator(0.1))
    axins2.tick_params(axis='x', top=True, labeltop=True, 
        bottom=False, labelbottom=False, which='both')
    axins2.set_yticks(np.arange(len(stellar_types)))
    axins2.set_yticklabels([])
    axins2.set_xlim(4.2, 4.8)
    axins2.set_ylim(-0.25, len(stellar_types)-0.75)
    axins2.plot(time, mapped_stellar_type, ds='steps-pre', c='k')
    ax2.indicate_inset_zoom(axins2, edgecolor='k')

    for i in range(len(stellar_types)):
        ax2.axhline(i, c='k', lw=1, ls=':')
        axins2.axhline(i, c='k', lw=1, ls=':')


    fig.subplots_adjust(hspace=0)
    fig.savefig('figures/gravo_stellar_timescale_evolution.pdf', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_timescale_evolution.png', bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='constant')
    parser.add_argument('--filepath', dest='filepath', type=str, default='./data/', required=False)
    args = parser.parse_args()

    np.random.seed(49023723)

    timescales = [0.1, 0.3, 1., 3., 10., 30., 100.] | units.kyr
    eta = np.array([0.01, 0.03, 0.1, 0.3, 1., 3., 10.])
    end_time = 6. | units.Myr

    N = 100

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 8.]|units.MSun,
        alphas=[-1.3, -2.3])

    mass = kroupa_imf.next_mass(N)

    if args.mode == 'constant' or args.mode == 'reversed':
        mass[-1] = 16. | units.MSun
    elif args.mode == 'adaptive':
        mass[-1] = 50. | units.MSun
        

    converter = nbody_system.nbody_to_si(mass.sum(), end_time)
    stars = new_plummer_model(N, converter)
    print ('Virial radius:', stars.virial_radius().value_in(units.pc), flush=True)
    print ('Plummer radius:', stars.virial_radius().value_in(units.pc)*3.*np.pi/16., flush=True)
    stars.mass = mass


    if args.mode == 'constant':
        #run_convergence_gravostellar(stars, timescales, end_time,   # run for output
        #    output_path=args.filepath)
        #run_convergence_gravostellar(stars, timescales, end_time)   # run for timings
        plot_convergence_gravostellar(args.filepath)
    elif args.mode == 'reversed':
        #run_convergence_gravostellar_reversed(stars, timescales, end_time)
        plot_convergence_gravostellar_reversed()
    elif args.mode == 'adaptive':
        #run_convergence_gravostellar_adaptive(stars, eta, end_time)
        plot_convergence_gravostellar_adaptive()
        plot_timescale_evolution_gravostellar_adaptive()


    plt.show()
