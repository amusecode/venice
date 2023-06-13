import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gs
import matplotlib.ticker as mpt
import h5py

import os

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.seba.interface import SeBa
from amuse.community.ph4.interface import ph4
#from amuse.community.galactics.interface import GalactICs
from amuse.community.bhtree.interface import BHTree
#from amuse.community.octgrav.interface import Octgrav
#from amuse.community.fi.interface import Fi
from amuse.community.gadget2.interface import Gadget2
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
    gravity = ph4(converter, number_of_workers=1)
    gravity.parameters.force_sync = True

    stellar = SeBa()

    system = Venice()
    system.record_runtime = True

    system.rest_order = 2
    system.cc_order = 2

    system.add_code(gravity)
    system.add_code(stellar)

    system.add_channel(1, 0, from_attributes=['mass'], to_attributes=['mass'])

    system.timescale_matrix[0,1] = timescale

    return system, gravity, stellar


def setup_gravity_stellar_r (timescale):

    converter = nbody_system.nbody_to_si(timescale, 1.|units.MSun)
    gravity = ph4(converter)
    gravity.parameters.force_sync = True

    stellar = SeBa()

    system = Venice()
    system.record_runtime = True

    system.rest_order = 2
    system.cc_order = 2

    system.add_code(stellar)
    system.add_code(gravity)

    system.add_channel(0, 1, from_attributes=['mass'], to_attributes=['mass'])

    system.timescale_matrix[0,1] = timescale

    return system, gravity, stellar


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
        stellar.particles.time_step.min() * eta

    return system, gravity, stellar


def get_energy_error (filepath, ics):

    E0 = ics.kinetic_energy() + ics.potential_energy()

    files = os.listdir(filepath)

    gravity_files = list(filter(lambda filename: (filename.split('_')[0] == 'plt')*(filename.split('_')[1] == 'gravity'), files))
    gravity_files.sort(key=lambda filename: filename.split('_')[3][1:])

    time = np.zeros(len(gravity_files)) | units.Myr
    energy_error = np.zeros(len(gravity_files))

    for i in range(len(gravity_files)):

        particles = read_set_from_file(filepath + gravity_files[i])

        time[i] = particles.get_timestamp()

        E = particles.kinetic_energy() + particles.potential_energy()

        energy_error[i] = np.abs((E - E0)/E0)

    print (filepath, 'done', flush=True)

    return time, energy_error


def get_star_trails (filepath, stars):

    files = os.listdir(filepath)

    gravity_files = list(filter(lambda filename: (filename.split('_')[0] == 'plt')*(filename.split('_')[1] == 'gravity'), files))
    gravity_files.sort(key=lambda filename: filename.split('_')[3][1:])

    x = np.zeros((len(gravity_files),len(stars))) | units.pc
    y = np.zeros((len(gravity_files),len(stars))) | units.pc
    z = np.zeros((len(gravity_files),len(stars))) | units.pc

    for i in range(len(gravity_files)):

        particles = read_set_from_file(filepath + gravity_files[i])

        x[i] = particles.x
        y[i] = particles.y
        z[i] = particles.z

    return x, y, z


def run_convergence_gravostellar (stars, timescales, end_time, orders=[1,2]):

    for i in range(len(orders)):

        stars_final = []

        timer_framework = np.zeros(len(timescales))
        timer_codes = np.zeros((len(timescales), 2))

        for j in range(len(timescales)):

            print (timescales[j].value_in(units.kyr), flush=True)

            system, gravity, stellar = setup_gravity_stellar(timescales[j])
            gravity.particles.add_particles(stars.copy())
            stellar.particles.add_particles(stars.copy())

            system.rest_order = orders[i]
            #system.verbose = True

            if j == 0:
                system.io_scheme = 2
                system.filepath = '/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_short_{a}/'.format(a=orders[i])
                system.save_data[0] = lambda code, filename: write_set_to_amuse_file(
                    code, filename, 'gravity')
            if j == len(timescales)-1:
                system.io_scheme = 2
                system.filepath = '/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_long_{a}/'.format(a=orders[i])
                system.save_data[0] = lambda code, filename: write_set_to_amuse_file(
                    code, filename, 'gravity')

            system.evolve_model(end_time)

            stars_final.append(gravity.particles.copy())

            print (system.runtime_framework, system.runtime_codes, flush=True)

            timer_framework[j] = system.runtime_framework
            timer_codes[j] = system.runtime_codes

            system.stop()

        np.savetxt('data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            np.array([timescales.value_in(units.kyr), timer_framework,
                timer_codes[:,0], timer_codes[:,1]]).T)


        dR_med = np.zeros(len(timescales)-1) | units.pc

        for j in range(len(timescales)-1):

            dR = (stars_final[j].position - stars_final[j+1].position).lengths()
            dR_med[j] = dR.median()

        t_mid = ((timescales[1:]*timescales[:-1])**0.5)

        np.savetxt('data/dR_med_gravostellar_order_{a}.txt'.format(a=orders[i]),
            np.array([t_mid.value_in(units.kyr), dR_med.value_in(units.pc)]).T)


def run_convergence_gravostellar_reversed (stars, timescales, end_time,
        orders=[1,2]):

    for i in range(len(orders)):

        stars_final = []
        stars_final_r = []

        timer_framework = np.zeros((2, len(timescales)))
        timer_codes = np.zeros((2, len(timescales), 2))

        dR_med = np.zeros(len(timescales)) | units.pc

        for j in range(len(timescales)):

            print (timescales[j].value_in(units.kyr), flush=True)

            system, gravity, stellar = setup_gravity_stellar(timescales[j])
            gravity.particles.add_particles(stars.copy())
            stellar.particles.add_particles(stars.copy())

            system.rest_order = orders[i]
            #system.verbose = True

            system_r, gravity_r, stellar_r = setup_gravity_stellar_r(timescales[j])
            gravity_r.particles.add_particles(stars.copy())
            stellar_r.particles.add_particles(stars.copy())

            system_r.rest_order = orders[i]
            #system_r.verbose = True

            system.evolve_model(end_time)
            system_r.evolve_model(end_time)

            stars_final.append(gravity.particles.copy())
            stars_final_r.append(gravity_r.particles.copy())

            print (system.runtime_framework, system.runtime_codes, flush=True)
            print (system_r.runtime_framework, system_r.runtime_codes, flush=True)

            timer_framework[0,j] = system.runtime_framework
            timer_codes[0,j] = system.runtime_codes

            timer_framework[1,j] = system.runtime_framework
            timer_codes[1,j] = system_r.runtime_codes

            dR = (stars_final[j].position - stars_final_r[j].position).lengths()
            dR_med[j] = dR.median()

            system.stop()

        np.savetxt('data/dR_med_gravostellar_reversed_order_{a}.txt'.format(
                a=orders[i]),
            np.array([timescales.value_in(units.kyr), dR_med.value_in(units.pc)]).T)

        np.savetxt('data/timer_gravostellar_reversed_order_{a}.txt'.format(
                a=orders[i]),
            np.array([timescales.value_in(units.kyr), 
                timer_framework[0], timer_codes[0,:,0], timer_codes[0,:,1],
                timer_framework[1], timer_codes[1,:,0], timer_codes[1,:,1]]).T)


def run_convergence_gravostellar_adaptive (stars, eta, end_time, orders=[1, 2]):

    for i in range(len(orders)):

        stars_final = []

        timer_framework = np.zeros(len(eta))
        timer_codes = np.zeros((len(eta), 2))

        for j in range(len(eta)):

            print (eta[j], flush=True)

            system, gravity, stellar = setup_gravity_stellar_adaptive(eta[j])
            gravity.particles.add_particles(stars)
            stellar.particles.add_particles(stars)

            system.rest_order = orders[i]
            system.record_timescales = True

            system.evolve_model(end_time)

            stars_final.append(gravity.particles.copy())

            timer_framework[j] = system.runtime_framework
            timer_codes[j] = system.runtime_codes

            np.savetxt('data/timescales_eta_{a}_order_{b}.txt'.format(a=eta[j], b=orders[i]), 
                system.timescales_codes[0].value_in(units.kyr))

            ind = (stars.mass - stars_final[j].mass).argmax()
            print (stars.mass[ind].value_in(units.MSun),
                stars_final[j].mass[ind].value_in(units.MSun), flush=True)
            print (timer_framework[j], timer_codes[j], flush=True)

            system.stop()

        np.savetxt('data/timer_gravostellar_adaptive_order_{a}.txt'.format(a=orders[i]),
            np.array([eta, timer_framework, timer_codes[:,0], timer_codes[:,1]]).T)


        dR_med = np.zeros(len(eta)-1) | units.pc

        for j in range(len(eta)-1):

            dR = (stars_final[j].position - stars_final[j+1].position).lengths()
            dR_med[j] = dR.median()

        eta_mid = (eta[1:]*eta[:-1])**0.5

        np.savetxt('data/dR_med_gravostellar_adaptive_order_{a}.txt'.format(a=orders[i]),
            np.array([eta_mid, dR_med.value_in(units.pc)]).T)


def plot_convergence_gravostellar (orders=[1, 2]):

    fig = plt.figure(figsize=(6.4, 9.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale [kyr]')
    ax1.set_ylabel('Median position error [pc]')

    ax1.plot(0., 0., c='C0', label='1$^\\mathrm{st}$ order')
    ax1.plot(0., 0., c='C1', label='2$^\\mathrm{nd}$ order')

    ax1.set_xlim(3e-2, 3e2)
    ax1.set_ylim(3e-10, 1e-5)

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Coupling timescale [kyr]')
    ax2.set_ylabel('Run-time [s]')

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax2.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax2.set_xlim(3e-2, 3e2)
    ax2.set_ylim(1e-2, 1e3)


    ls = ['--', '-.']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        t_mid, dR_med = np.loadtxt(
            'data/dR_med_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        ax1.errorbar(t_mid, dR_med,
            xerr=(t_mid-timescales[:-1], timescales[1:]-t_mid),
            capsize=3., fmt='none', c='C'+str(i))

        ax1.plot(t_mid, dR_med[-1] * (t_mid/t_mid[-1])**orders[i],
            c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{'+str(orders[i])+'}'))

        ax2.plot(timescales, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax2.plot(timescales, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax2.plot(timescales, timer_stellar, 
            linestyle=':', c='C'+str(i))

    #ax1.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)
    #ax2.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    fig.subplots_adjust(hspace=0)

    fig.savefig('figures/gravo_stellar_constant_convergence.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_constant_convergence.pdf', bbox_inches='tight')


def plot_convergence_gravostellar_v2 (orders=[1, 2]):

    fig = plt.figure(figsize=(12.8, 4.8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale [kyr]')
    ax1.set_ylabel('Median position error [pc]')

    ax1.plot(0., 0., c='C0', label='1$^\\mathrm{st}$ order')
    ax1.plot(0., 0., c='C1', label='2$^\\mathrm{nd}$ order')

    ax1.set_xlim(3e-2, 3e2)
    ax1.set_ylim(1e-10, 1e-5)

    ax1.set_xticks([1e-1, 1e0, 1e1, 1e2])


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Run-time [s]')
    ax2.set_ylabel('Median position error [pc]')

    ax2.set_xlim(3e-1, 1e3)
    ax2.set_ylim(1e-10, 1e-5)

    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    ax2.set_xticks([1e0, 1e1, 1e2, 1e3])


    ls = ['--', '-.']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        t_mid, dR_med = np.loadtxt(
            'data/dR_med_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        ax1.errorbar(t_mid, dR_med,
            xerr=(t_mid-timescales[:-1], timescales[1:]-t_mid),
            capsize=3., fmt='C{a},:'.format(a=i))

        ax1.plot(t_mid, dR_med[len(t_mid)//2] * (t_mid/t_mid[len(t_mid)//2])**orders[i],
            c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{'+str(orders[i])+'}'))

        timer = timer_framework + timer_gravity + timer_stellar
        timer_mid = (timer[1:]*timer[:-1])**0.5

        ax2.errorbar(timer_mid, dR_med,
            xerr=(timer_mid-timer[1:], np.abs(timer[:-1]-timer_mid)),
            capsize=3., fmt='C{a},:'.format(a=i))

        #ax2.plot(timer_mid, dR_med[0] * (timer_mid/timer_mid[0])**-orders[i],
        #    c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{-'+str(orders[i])+'}'))

        if i == 1:
            ax2.text(timer[0], dR_med[0]/1.2, 'dt={b}-{a} kyr'.format(a=timescales[0], b=timescales[1]), ha='right', va='top')
        if i == 0:
            ax2.text(timer[-1], 1.2*dR_med[-1], 'dt={a}-{b} kyr'.format(a=str(timescales[-1])[:-2], b=str(timescales[-2])[:-2]))

    ax1.legend(frameon=False)
    #ax2.legend(frameon=False)

    fig.subplots_adjust(wspace=0)

    fig.savefig('figures/gravo_stellar_constant_convergence_v2.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_constant_convergence_v2.pdf', bbox_inches='tight')


def plot_convergence_gravostellar_v3 (stars, orders=[1, 2]):

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
    '''
    ax4.set_xlabel('Time [Myr]')
    ax4.set_ylabel('|E(t) - E(0)|/E(0)')

    

    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()

    ax4.set_xlim(0., 6.)
    ax4.set_ylim(3e-7, 3e-3)

    ax4.xaxis.set_minor_locator(mpt.MultipleLocator(0.2))
    '''


    ls = ['--', ':']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        t_mid, dR_med = np.loadtxt(
            'data/dR_med_gravostellar_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        ax1.errorbar(t_mid, dR_med,
            xerr=(t_mid-timescales[:-1], timescales[1:]-t_mid),
            capsize=3., fmt='C{a},:'.format(a=i))

        ax1.plot(t_mid, dR_med[len(t_mid)//2] * (t_mid/t_mid[len(t_mid)//2])**orders[i],
            c='k', ls=ls[i])#, label='$dR\\sim\\tau^{a}$'.format(a='{'+str(orders[i])+'}'))
        if i == 0:
            ax1.text(t_mid[-1], dR_med[-1]*1.2, 'dR~dt$^1$')
        if i == 1:
            ax1.text(t_mid[-1], dR_med[-1]/1.2, 'dR~dt$^2$', va='top')

        timer = timer_framework + timer_gravity + timer_stellar
        timer_mid = (timer[1:]*timer[:-1])**0.5

        ax2.errorbar(timer_mid, dR_med,
            xerr=(timer_mid-timer[1:], np.abs(timer[:-1]-timer_mid)),
            capsize=3., fmt='C{a},:'.format(a=i))

        #ax2.plot(timer_mid, dR_med[0] * (timer_mid/timer_mid[0])**-orders[i],
        #    c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{-'+str(orders[i])+'}'))

        if i == 1:
            ax2.text(timer[0], dR_med[0]/1.2, 'dt={b}-{a} kyr'.format(a=timescales[0], b=timescales[1]), ha='right', va='top')
        if i == 0:
            ax2.text(timer[-1], 1.2*dR_med[-1], 'dt={a}-{b} kyr'.format(a=str(timescales[-1])[:-2], b=str(timescales[-2])[:-2]))


        ax3.plot(timescales, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax3.plot(timescales, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax3.plot(timescales, timer_stellar, 
            linestyle=':', c='C'+str(i))

        x_short, y_short, z_short = get_star_trails('/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_short_{a}/'.format(a=orders[i]), stars)
        x_long, y_long, z_long = get_star_trails('/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_long_{a}/'.format(a=orders[i]), stars)

        dR = ( (x_short[::16] - x_long)**2. + (y_short[::16] - y_long)**2. + (z_short[::16] - z_long)**2. )**0.5
        ax4.plot(dR.value_in(units.pc), c='C'+str(i), ls='--')

        #ax4.plot(np.abs((x_short[::16]-x_long).value_in(units.pc)), c='C'+str(i), ls='--')
        #ax4.plot(x_long.value_in(units.pc), y_long.value_in(units.pc), c='C'+str(i), ls=':')

        '''
        time_short, energy_error_short = get_energy_error(
            '/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_short_{a}/'.format(a=orders[i]), stars)
        time_long, energy_error_long = get_energy_error(
            '/data2/wilhelm/sim_archive/venice/gravo_stellar_constant_long_{a}/'.format(a=orders[i]), stars)

        ax4.plot(time_short.value_in(units.Myr), energy_error_short, c='C'+str(i), ls='--')
        ax4.plot(time_long.value_in(units.Myr), energy_error_long, c='C'+str(i), ls=':')

        if i == 0:
            ax4.plot([0.], [0.], c='k', ls='--', label='dt={a} kyr'.format(a=timescales[0]))
            ax4.plot([0.], [0.], c='k', ls=':', label='dt={a} kyr'.format(a=timescales[-1]))
        '''


    ax1.legend(frameon=False)
    ax3.legend(frameon=False)
    #ax4.legend(frameon=False)

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('figures/gravo_stellar_constant_convergence_v3.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_constant_convergence_v3.pdf', bbox_inches='tight')


def plot_convergence_gravostellar_reversed ():

    orders = [1, 2]


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

    ax1.set_xlim(3e-1, 3e2)
    ax1.set_ylim(1e-9, 1e-5)


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Coupling timescale [kyr]')
    ax2.set_ylabel('Run-time [s]')

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax2.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax2.set_xlim(3e-1, 3e2)
    ax2.set_ylim(1e-2, 3e3)


    ls = ['--', '-.']
    for i in range(len(orders)):

        timescales, dR_med = np.loadtxt(
            'data/dR_med_gravostellar_reversed_order_{a}.txt'.format(a=orders[i]),
            unpack=True)
        _, timer_framework, timer_gravity, timer_stellar, timer_framework_r, \
            timer_stellar_r, timer_gravity_r = np.loadtxt(
            'data/timer_gravostellar_reversed_order_{a}.txt'.format(a=orders[i]),
            unpack=True)

        ax1.scatter(timescales, dR_med, c='C'+str(i))

        ax1.plot(timescales, dR_med[-1] * (timescales/timescales[-1])**orders[i],
            c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{'+str(orders[i])+'}'))

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

    #ax1.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)
    #ax2.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)

    fig.subplots_adjust(hspace=0)


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

    ax1.set_xlim(3e-2, 3e1)
    ax1.set_ylim(1e-4, 1e-2)

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()


    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Timescale parameter $\\eta$')
    ax2.set_ylabel('Run-time [s]')

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Gravity')
    ax2.plot(0., 0., linestyle=':', c='k', label='Stellar')

    ax2.set_xlim(3e-2, 3e1)
    ax2.set_ylim(3e-2, 1e2)


    ls = ['--', '-.']
    for i in range(len(orders)):

        timescales, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
            'data/timer_gravostellar_adaptive_order_{a}.txt'.format(a=orders[i]),
            #'data/timer_gravostellar_adaptive.txt',
            unpack=True)

        t_mid, dR_med = np.loadtxt(
            'data/dR_med_gravostellar_adaptive_order_{a}.txt'.format(a=orders[i]),
            #'data/dR_med_gravostellar_adaptive.txt',
            unpack=True)

        ax1.errorbar(t_mid, dR_med,
            xerr=(t_mid-timescales[:-1], timescales[1:]-t_mid),
            capsize=3., fmt='none', c='C'+str(i))

        ax1.plot(t_mid, dR_med[len(t_mid)//2] * (t_mid/t_mid[len(t_mid)//2])**orders[i],
            c='k', ls=ls[i], label='$dR\\sim\\tau^{a}$'.format(a='{'+str(orders[i])+'}'))

        ax2.plot(timescales, timer_framework, 
            linestyle='-', c='C'+str(i))
        ax2.plot(timescales, timer_gravity, 
            linestyle='--', c='C'+str(i))
        ax2.plot(timescales, timer_stellar, 
            linestyle=':', c='C'+str(i))

    #ax1.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)
    #ax2.set_xlim(np.min(timescales)/3., np.max(timescales)*3.)

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    fig.subplots_adjust(hspace=0)

    fig.savefig('figures/gravo_stellar_adaptive_convergence.png', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_adaptive_convergence.pdf', bbox_inches='tight')


def plot_timescale_evolution_gravostellar_adaptive ():

    eta, timer_framework, timer_gravity, timer_stellar = np.loadtxt(
        'data/timer_gravostellar_adaptive_order_1.txt', unpack=True)

    stellar = SeBa()
    stellar.particles.add_particles(Particles(1, mass=50.|units.MSun))

    stellar_type = [stellar.particles[0].stellar_type.value_in(units.stellar_type)]
    time = [0.]

    t = 0.|units.Myr
    eta_min = np.min(eta)

    timescales = np.loadtxt('data/timescales_eta_{a}_order_1.txt'.format(a=eta_min))
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

    fig = plt.figure(figsize=(6.4, 9.6))
    grd = gs.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[1., 0.6])
    ax1 = fig.add_subplot(grd[0])
    ax2 = fig.add_subplot(grd[1])

    axins1 = ax1.inset_axes([0.1, 0.1, 0.55, 0.7])
    axins1.set_xticks([4.2, 4.5, 4.8])
    axins1.set_yticklabels([])
    axins1.xaxis.set_minor_locator(mpt.MultipleLocator(0.1))
    axins1.set_xlim(4.2, 4.8)
    axins1.set_ylim(1e2, 1e5)
    axins1.set_yscale('log')

    counter = 0
    for i in range(len(eta)):
        if eta[i] >= 1.:
            timescales = np.loadtxt('data/timescales_eta_{a}_order_1.txt'.format(a=eta[i]))
            ax1.plot(np.cumsum(timescales)/1e3, timescales*1e3, 
                ds='steps-pre', c='C'+str(counter), label='$\\eta$='+str(eta[i]))
            if eta[i] >= 1.:
                axins1.plot(np.cumsum(timescales)/1e3, timescales*1e3, 
                    ds='steps-pre', c='C'+str(counter))
            counter += 1

    ax1.set_yscale('log')

    ax1.set_xlabel('Time [Myr]')
    ax1.set_ylabel('Coupling timescale [yr]')

    ax1.set_xlim(0., 6.)
    ax1.set_ylim(1e-4, 1e6)

    ax1.tick_params(axis='x', bottom=False, labelbottom=False, 
        top=True, labeltop=True, which='both')
    ax1.xaxis.set_label_position('top')

    ax1.xaxis.set_minor_locator(mpt.MultipleLocator(0.2))

    ax1.legend(loc='lower right', frameon=False, fontsize=10)

    ax1.indicate_inset_zoom(axins1, edgecolor='k')


    #ax2.plot(time, stellar_type, ds='steps-pre', c='k')

    ax2.set_xlabel('Time [Myr]')
    ax2.set_ylabel('Stellar type')

    stellar_types = np.unique(stellar_type)
    labels = [ break_string(str(stype|units.stellar_type)) for stype in stellar_types ]

    mapped_stellar_type = np.zeros(len(stellar_type))
    for i in range(len(stellar_types)):
        mapped_stellar_type[ stellar_type == stellar_types[i] ] = i

    ax2.plot(time, mapped_stellar_type, ds='steps-pre', c='k')

    #ax2.set_yticks(stellar_types)
    ax2.set_yticks(np.arange(len(stellar_types)))
    ax2.set_yticklabels(labels, ha='left', va='bottom', fontsize=10)
    ax2.tick_params(axis='y', pad=-10)

    ax2.set_xlim(0., 6.)
    ax2.set_ylim(-0.5, len(stellar_types)-0.5)

    ax2.xaxis.set_minor_locator(mpt.MultipleLocator(0.2))

    axins2 = ax2.inset_axes([0.33, 0.1, 0.35, 0.7])
    axins2.set_xticks([4.2, 4.5, 4.8])
    axins2.xaxis.set_minor_locator(mpt.MultipleLocator(0.1))
    axins2.tick_params(axis='x', top=True, labeltop=True, 
        bottom=False, labelbottom=False, which='both')
    #axins2.set_yticks(stellar_types)
    axins2.set_yticks(np.arange(len(stellar_types)))
    axins2.set_yticklabels([])
    axins2.set_xlim(4.2, 4.8)
    axins2.set_ylim(-0.25, len(stellar_types)-0.75)
    axins2.plot(time, mapped_stellar_type, ds='steps-pre', c='k')
    ax2.indicate_inset_zoom(axins2, edgecolor='k')


    fig.subplots_adjust(hspace=0)
    fig.savefig('figures/gravo_stellar_timescale_evolution.pdf', bbox_inches='tight')
    fig.savefig('figures/gravo_stellar_timescale_evolution.png', bbox_inches='tight')


if __name__ == '__main__':

    np.random.seed(49023723)

    #'''
    timescales = [0.1, 0.3, 1., 3., 10., 30., 100.] | units.kyr
    #eta = np.array([0.03, 0.1, 0.3, 1., 3.])
    eta = np.array([0.1, 0.3, 1., 3., 10.])
    end_time = 6. | units.Myr

    N = 100
    R = 1. | units.pc

    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 8.]|units.MSun,
        alphas=[-1.3, -2.3])

    mass = kroupa_imf.next_mass(N)
    mass[-1] = 16. | units.MSun
    #mass[-1] = 50. | units.MSun

    converter = nbody_system.nbody_to_si(mass.sum(), end_time)
    stars = new_plummer_model(N, converter)
    print ('Virial radius:', stars.virial_radius().value_in(units.pc), flush=True)
    print ('Plummer radius:', stars.virial_radius().value_in(units.pc)*3.*np.pi/16., flush=True)
    stars.mass = mass

    #run_convergence_gravostellar(stars, timescales[4:], end_time)
    #plot_convergence_gravostellar()
    #plot_convergence_gravostellar_v2()
    plot_convergence_gravostellar_v3(stars)

    #run_convergence_gravostellar_reversed(stars, timescales[2:], end_time)
    #plot_convergence_gravostellar_reversed()

    #stars[-1].mass = 50. | units.MSun
    #run_convergence_gravostellar_adaptive(stars, eta, end_time)
    #plot_convergence_gravostellar_adaptive(orders=[1,2])
    #plot_timescale_evolution_gravostellar_adaptive()

    plt.show()
