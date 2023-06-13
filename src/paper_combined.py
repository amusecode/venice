import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gs
import matplotlib.ticker as tck
import h5py
import scipy.stats as ss

import argparse
import os
import time

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.seba.interface import SeBa
from amuse.community.ph4.interface import ph4
from amuse.community.gadget2.interface import Gadget2
from amuse.community.phantom.interface import Phantom
from amuse.community.fi.interface import Fi
from amuse.io import write_set_to_file, read_set_from_file
from amuse.ext.galactic_potentials import MWpotentialBovy2015, NFW_profile

from venice import Venice, DynamicKick, dynamic_kick
from test_separate import convert_magi_to_amuse


plt.rcParams.update({'font.size': 12})


def clear_output_directory (output_path):

    os.system('rm -f {a}/*.venice'.format(a=output_path))


def minimum_mutual_freefall_time (grav1, grav2, dt, direction='11'):

    dt_new = -dt

    grav1_particles = grav1.particles.copy(
        filter_attributes=lambda p, attribute_name: \
            attribute_name in ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius'])
    grav2_particles = grav2.particles.copy(
        filter_attributes=lambda p, attribute_name: \
            attribute_name in ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius'])

    if direction[0] != '0':

        ax, ay, az = grav1.get_gravity_at_point(grav2_particles.radius,
            grav2_particles.x, grav2_particles.y, grav2_particles.z)

        nearest_neighbours = grav2_particles.nearest_neighbour(grav1_particles)

        tau_ff = ( ((grav2_particles.position - nearest_neighbours.position)**2.).sum(axis=1) / \
            (ax*ax + ay*ay + az*az) )**0.25

        if dt_new < 0.*dt or dt_new > tau_ff.min():
            dt_new = tau_ff.min()

    if direction[1] != '0':

        ax, ay, az = grav2.get_gravity_at_point(grav1_particles.radius,
            grav1_particles.x, grav1_particles.y, grav1_particles.z)

        nearest_neighbours = grav1_particles.nearest_neighbour(grav2_particles)

        tau_ff = ( ((grav1_particles.position - nearest_neighbours.position)**2.).sum(axis=1) / \
            (ax*ax + ay*ay + az*az) )**0.25

        if dt_new < 0.*dt or dt_new > tau_ff.min():
            dt_new = tau_ff.min()

    if dt_new < 0.*dt:
        dt_new = 1e6 | units.Gyr

    return dt_new


def minimum_freefall_time_in_potential (potential, grav, dt):

    grav_particles = grav.particles.copy(
        filter_attributes=lambda p, attribute_name: \
            attribute_name in ['mass', 'x', 'y', 'z', 'radius'])

    ax, ay, az = potential.get_gravity_at_point(
        grav_particles.radius, grav_particles.x, grav_particles.y, grav_particles.z)

    tau_ff = ( grav_particles.position.lengths()**2/(ax*ax + ay*ay + az*az) )**0.25

    return tau_ff.min()


def setup_combined_system (params):

    converter_galaxy = nbody_system.nbody_to_si(1e12|units.MSun, 1.|units.kpc)
    converter_cluster = nbody_system.nbody_to_si(1e3|units.MSun, 1.|units.pc)

    system = Venice()

    gravity_galaxy = MWpotentialBovy2015()
    system.add_code(gravity_galaxy)

    gravity_cluster = ph4(converter_cluster)
    gravity_cluster.parameters.force_sync = True
    system.add_code(gravity_cluster)

    system.kick[0][1] = dynamic_kick
    system.update_timescale[0][1] = lambda g1, g2, dt: params['eta_gravity']*minimum_freefall_time_in_potential(
        g1, g2, dt)

    system.io_scheme = 2
    system.filepath = params['output_path']
    system.save_data[1] = lambda code, filename: write_set_to_file(
        code.particles, filename.format(code_label='cluster', set_label='particles'), 
        'hdf5', overwrite_file=True, timestamp=code.model_time)

    codes = [gravity_galaxy, gravity_cluster]

    if params['perturbers'] is not None:
        gravity_perturbers = ph4(converter_galaxy)
        gravity_perturbers.parameters.force_sync = True
        system.add_code(gravity_perturbers)
        code_id = len(system.codes)-1
        system.kick[code_id][1] = DynamicKick(radius_is_eps=True)
        system.kick[0][code_id] = DynamicKick(radius_is_eps=True)
        system.update_timescale[0][code_id] = lambda g1, g2, dt: params['eta_gravity']*minimum_freefall_time_in_potential(
            g1, g2, dt)
        system.update_timescale[1][code_id] = lambda g1, g2, dt: params['eta_gravity']*minimum_mutual_freefall_time(
            g1, g2, dt, direction='01')
        system.save_data[-1] = lambda code, filename: write_set_to_file(
        code.particles, filename.format(code_label='perturbers', set_label='particles'), 
            'hdf5', overwrite_file=True, timestamp=code.model_time)
        codes.append(gravity_perturbers)

    if params['stellar_evolution']:
        stellar = SeBa()
        system.add_code(stellar)
        code_id = len(system.codes)-1
        system.add_channel(code_id, 1, 
            from_attributes=['mass'], to_attributes=['mass'])
        system.update_timescale[1][code_id] = lambda g, s, dt: params['eta_stellar']*s.particles.time_step.min()
        system.save_data[-1] = lambda code, filename: write_set_to_file(
        code.particles, filename.format(code_label='stellar', set_label='particles'), 
            'hdf5', overwrite_file=True, timestamp=code.model_time)
        codes.append(stellar)
        
    return system, codes


def setup_initial_conditions (codes, params):

    # CLUSTER
    kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 8.]|units.MSun,
        alphas=[-1.3, -2.3])
    mass = kroupa_imf.next_mass(params['N_cluster'])
    if params['max_mass_cluster'] is not None:
        mass[:len(params['max_mass_cluster'])] = params['max_mass_cluster']

    converter = nbody_system.nbody_to_si(mass.sum(), params['R_cluster'])
    cluster = new_plummer_model(params['N_cluster'], converter)
    cluster.mass = mass
    cluster.radius = params['eps_cluster']

    cluster.x += params['dR_cluster']
    cluster_com = cluster.center_of_mass()
    ax, ay, az = MWpotentialBovy2015().get_gravity_at_point(0.|units.pc, cluster_com[0], cluster_com[1], cluster_com[2])
    a2 = ax*ax + ay*ay + az*az
    v2_kep = cluster_com.length()*a2**0.5
    cluster.vy += ((1.-params['e_cluster'])/(1.+params['e_cluster'])*v2_kep)**0.5

    codes[1].particles.add_particles(cluster)
    if params['stellar_evolution']:
        codes[2+(params['perturbers'] is not None)].particles.add_particles(cluster)


    # PERTURBERS
    if params['perturbers'] is not None:
        params['perturbers'].x = params['perturbers'].semimajor_axis * np.cos(params['perturbers'].argument_of_periastron)
        params['perturbers'].y = params['perturbers'].semimajor_axis * np.sin(params['perturbers'].argument_of_periastron)
        params['perturbers'].z = 0. | units.kpc
        params['perturbers'].velocity = [0., 0., 0.] | units.kms
        for i in range(len(params['perturbers'])):
            e = params['perturbers'][i].eccentricity
            ax, ay, az = MWpotentialBovy2015().get_gravity_at_point(0.|units.pc, 
                params['perturbers'][i].x, params['perturbers'][i].y, params['perturbers'][i].z)
            a2 = ax*ax + ay*ay + az*az
            v2_kep = params['perturbers'][i].position.length()*a2**0.5
            v = ((1.-e)/(1.+e)*v2_kep)**0.5
            params['perturbers'][i].vx = -v * np.sin(params['perturbers'][i].argument_of_periastron)
            params['perturbers'][i].vy =  v * np.cos(params['perturbers'][i].argument_of_periastron)
        codes[2].particles.add_particles(params['perturbers'])


def run_combined_model (params):

    clear_output_directory(params['output_path'])

    system, codes = setup_combined_system(params)

    setup_initial_conditions(codes, params)

    system.verbose = True

    system.evolve_model(params['end_time'])


def plot_combined_model (filepath):

    filepaths = [filepath + '/combined_s0_p0/', 
        filepath + '/combined_s1_p0/', 
        filepath + '/combined_s0_p1/', 
        filepath + '/combined_s1_p1/'
    ]

    N_panels = 1

    figs = [ plt.figure(), plt.figure(figsize=(12.8, 4.8)), plt.figure(), plt.figure(figsize=(6.4, 9.6)) ]
    ax1 = figs[0].add_subplot(111)
    ax2a = figs[1].add_subplot(121, aspect='equal')
    ax2b = figs[1].add_subplot(122, aspect='equal')
    ax3 = figs[2].add_subplot(111)
    ax4a = figs[3].add_subplot(211)
    ax4b = figs[3].add_subplot(212)


    ax1.set_yscale('log')

    ax1.set_xlabel('t [Myr]')
    ax1.set_ylabel('dt [yr]')

    ax1.plot([0.], [0.], label='Cluster', c='k', ls='--')
    ax1.plot([0.], [0.], label='Perturbers', c='k', ls='-.')
    ax1.plot([0.], [0.], label='Stellar', c='k', ls=':')

    ax1.legend(frameon=False)

    ax1.set_xlim(0., 5000.)
    ax1.set_ylim(1e0, 3e8)

    ax1.xaxis.set_minor_locator(tck.MultipleLocator(250.))


    ax2a.set_xlabel('x [kpc]')
    ax2a.set_ylabel('y [kpc]')
    ax2b.set_xlabel('x [kpc]')
    ax2b.set_ylabel('y [kpc]')

    ax2a.set_xlim(-100., 100.)
    ax2a.set_ylim(-100., 100.)
    ax2b.set_xlim(-10., 10.)
    ax2b.set_ylim(-10., 10.)

    ax2a.plot([0.], [0.], c='k', label='perturbers')
    ax2a.legend(loc='upper right', frameon=False)

    ax2b.scatter([-100.], [-100.], c='C0', label='None')#, s=1.)
    ax2b.scatter([-100.], [-100.], c='C1', label='Stellar')#, s=1.)
    ax2b.scatter([-100.], [-100.], c='C2', label='Perturbers')#, s=1.)
    ax2b.scatter([-100.], [-100.], c='C3', label='Both')#, s=1.)
    ax2b.legend(loc='upper left', frameon=False)

    ax2a.xaxis.set_minor_locator(tck.MultipleLocator(10.))
    ax2a.yaxis.set_minor_locator(tck.MultipleLocator(10.))
    ax2b.xaxis.set_minor_locator(tck.MultipleLocator(1.))
    ax2b.yaxis.set_minor_locator(tck.MultipleLocator(1.))


    ax3.set_xlabel('$\\phi$ [rad]')
    ax3.set_ylabel('N')

    ax3.plot([0.], [0.], label='None', c='C0')
    ax3.plot([0.], [0.], label='Stellar', c='C1')
    ax3.plot([0.], [0.], label='Perturbers', c='C2')
    ax3.plot([0.], [0.], label='Both', c='C3')

    ax3.legend(frameon=False)


    ax4a.set_xlabel('t [Myr]')
    ax4a.set_ylabel('$\\sigma_R$/$\\mu_R$')

    ax4a.set_yscale('log')

    ax4a.plot([-100.], [-100.], c='C0', label='None')
    ax4a.plot([-100.], [-100.], c='C1', label='Stellar')
    ax4a.plot([-100.], [-100.], c='C2', label='Perturbers')
    ax4a.plot([-100.], [-100.], c='C3', label='Both')
    ax4a.legend(loc='upper left', frameon=False)

    ax4a.set_xlim(0., 5000.)

    ax4b.set_xlabel('t [Myr]')
    ax4b.set_ylabel('$\\sigma_{\\phi}$ [rad]')

    ax4b.set_xlim(0., 5000.)
    ax4b.set_ylim(0., np.pi)

    ax4a.xaxis.set_minor_locator(tck.MultipleLocator(250.))
    ax4b.xaxis.set_minor_locator(tck.MultipleLocator(250.))



    for i in range(len(filepaths)):

        files = os.listdir(filepaths[i])

        #*(int(filename.split('_')[3][1:-7])%30 == 0)
        cluster_files = list(filter(lambda filename: (filename.split('_')[0] == 'plt')*(filename.split('_')[1] == 'cluster'), files))
        cluster_files.sort(key=lambda filename: filename.split('_')[3][1:])

        if len(cluster_files) > 0:

            time = np.zeros(len(cluster_files)) | units.Myr
            dR = np.zeros(len(cluster_files))
            dphi = np.zeros(len(cluster_files))

            for j in range(len(cluster_files)):

                particles = read_set_from_file(filepaths[i] + cluster_files[j])
                time[j] = particles.get_timestamp()
                R = particles.position.lengths()
                dR[j] = R.std()/R.mean()
                phi = np.arctan2(particles.y.value_in(units.kpc), particles.x.value_in(units.kpc))
                # Center on 0, so there's no splitting a peak between -pi and pi
                phi -= np.max(phi)
                phi[ phi < -np.pi] += 2.*np.pi
                dphi[j] = np.std(phi)

            if i == 3:
                ax1.plot(time[1:].value_in(units.Myr), np.diff(time.value_in(units.yr)), ds='steps-pre', ls='--', c='C'+str(i))

            ax2a.scatter(particles.x.value_in(units.kpc), particles.y.value_in(units.kpc), s=1., c='C'+str(i))
            ax2b.scatter(particles.x.value_in(units.kpc), particles.y.value_in(units.kpc), s=1., c='C'+str(i))

            phi = np.arctan2(particles.y.value_in(units.kpc), particles.x.value_in(units.kpc))
            kde = ss.gaussian_kde(phi)
            bins = np.linspace(-1., 1., num=300)*np.pi
            ax3.plot(bins, kde(bins), c='C'+str(i))
            #ax3.hist(phi, 
            #    bins=np.linspace(-1., 1., num=30)*np.pi, histtype='step', color='C'+str(i))
            #ax3.axvline(np.quantile(phi, 0.16), color='C'+str(i), linestyle='-.')
            #ax3.axvline(np.quantile(phi, 0.50), color='C'+str(i), linestyle='--')
            #ax3.axvline(np.quantile(phi, 0.84), color='C'+str(i), linestyle='-.')
            print (np.std(phi))

            ax4a.plot(time.value_in(units.Myr), dR, c='C'+str(i))
            ax4b.plot(time.value_in(units.Myr), dphi, c='C'+str(i))

            print (filepaths[i], "cluster", flush=True)


        perturbers_files = list(filter(lambda filename: (filename.split('_')[0] == 'plt')*(filename.split('_')[1] == 'perturbers'), files))
        perturbers_files.sort(key=lambda filename: filename.split('_')[3][1:])

        if len(perturbers_files) > 0:

            time = np.zeros(len(perturbers_files)) | units.Myr
            R_perturbers = np.zeros((len(perturbers_files), 2, 3)) | units.kpc

            for j in range(len(perturbers_files)):

                particles = read_set_from_file(filepaths[i] + perturbers_files[j])
                time[j] = particles.get_timestamp()
                R_perturbers[j] = particles.position

            if i == 3:
                ax1.plot(time[1:].value_in(units.Myr), np.diff(time.value_in(units.yr)), ds='steps-pre', ls='-.', c='C'+str(i))

            ax2a.plot(R_perturbers[:,0,0].value_in(units.kpc), R_perturbers[:,0,1].value_in(units.kpc), c='k', ls='--')
            ax2a.plot(R_perturbers[:,1,0].value_in(units.kpc), R_perturbers[:,1,1].value_in(units.kpc), c='k', ls=':')

            print (filepaths[i], "perturbers", flush=True)


        stellar_files = list(filter(lambda filename: (filename.split('_')[0] == 'plt')*(filename.split('_')[1] == 'stellar'), files))
        stellar_files.sort(key=lambda filename: filename.split('_')[3][1:])

        if len(stellar_files) > 0:

            time = np.zeros(len(stellar_files)) | units.Myr

            for j in range(len(stellar_files)):

                particles = read_set_from_file(filepaths[i] + stellar_files[j])
                time[j] = particles.get_timestamp()

            if i == 3:
                ax1.plot(time[1:].value_in(units.Myr), np.diff(time.value_in(units.yr)), ds='steps-pre', ls=':', c='C'+str(i))

            print (filepaths[i], "stellar", flush=True)

    figs[0].savefig('figures/combined_timesteps.pdf', bbox_inches='tight')
    figs[0].savefig('figures/combined_timesteps.png', bbox_inches='tight')

    figs[1].savefig('figures/combined_positions.pdf', bbox_inches='tight')
    figs[1].savefig('figures/combined_positions.png', bbox_inches='tight')

    figs[3].savefig('figures/combined_coordinates_std.pdf', bbox_inches='tight')
    figs[3].savefig('figures/combined_coordinates_std.png', bbox_inches='tight')


if __name__ == '__main__':

    np.random.seed(42937523)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stellar-evolution', dest='stellar_evolution', type=int, default=0)
    parser.add_argument('-p', '--perturbers', dest='perturbers', type=int, default=0)
    parser.add_argument('--filepath', dest='filepath', type=str, default='./data/', required=False)

    args = parser.parse_args()

    perturbers = Particles(2,
        mass = 1e11 | units.MSun,
        radius = 1. | units.kpc,
        eccentricity = [0., 0.],
        semimajor_axis = [60., 50.] | units.kpc,
        argument_of_periastron = [np.pi/2., np.pi]
    )

    params = {
        'end_time': 5000. | units.Myr,
        'output_path': args.filepath + '/combined_s{a}_p{b}/'.format(
            a='1' if args.stellar_evolution else '0',
            b='1' if args.perturbers else '0'),
        #'eps_galaxy': 1. | units.pc,
        'eta_gravity': 0.03,
        'eta_stellar': 0.3,
        'N_cluster': 100,
        'R_cluster': 10. | units.pc,
        'eps_cluster': 1. | units.pc,
        'dR_cluster': 5. | units.kpc,
        'e_cluster': 0.,
        'max_mass_cluster': [50., 30., 16.] | units.MSun,
        'perturbers': None,
        'stellar_evolution': args.stellar_evolution == True,
    }

    if args.perturbers:
        params['perturbers'] = perturbers

    #run_combined_model(params)

    if args.perturbers and args.stellar_evolution:
        plot_combined_model(args.filepath)

    plt.show()
