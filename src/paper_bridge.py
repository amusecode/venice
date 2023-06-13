import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import argparse

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.ph4.interface import ph4
from amuse.io import write_set_to_file, read_set_from_file
from amuse.ext.galactic_potentials import MWpotentialBovy2015

from venice import Venice, DynamicKick, dynamic_kick


plt.rcParams.update({'font.size': 12})


kroupa_imf = MultiplePartIMF(mass_boundaries=[0.08, 0.5, 8.]|units.MSun,
    alphas=[-1.3, -2.3])


def setup_bridged_gravity (code, timescale, converter):

    gravity1 = code(converter)
    gravity2 = MWpotentialBovy2015()

    system = Venice()

    system.add_code(gravity1)
    system.add_code(gravity2)

    system.kick[1][0] = dynamic_kick
    system.timescale_matrix[0,1] = timescale

    return system, gravity1, gravity2


def setup_bridged_gravity_adaptive (code, eta, converter):

    gravity1 = code(converter)
    gravity2 = MWpotentialBovy2015()

    system = Venice()

    system.add_code(gravity1)
    system.add_code(gravity2)

    system.kick[1][0] = dynamic_kick

    def minimum_freefall_time_in_potential (potential, grav, dt):

        grav_particles = grav.particles.copy(
            filter_attributes=lambda p, attribute_name: \
                attribute_name in ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'radius'])

        ax, ay, az = potential.get_gravity_at_point(
            grav_particles.radius, grav_particles.x, grav_particles.y, grav_particles.z)

        tau_ff = ( grav_particles.position.lengths()**2/(ax*ax + ay*ay + az*az) )**0.25

        return eta * tau_ff.min()

    system.update_timescale[1][0] = minimum_freefall_time_in_potential

    return system, gravity1, gravity2


def generate_cluster (N, radius, max_mass=None):

    mass = kroupa_imf.next_mass(N)
    if max_mass is not None:
        mass[-1] = max_mass

    converter = nbody_system.nbody_to_si(mass.sum(), radius)
    cluster = new_plummer_model(N, converter)
    cluster.mass = mass

    return cluster


def run_convergence_bridge_constant (cluster, timescales, end_time):

    converter = nbody_system.nbody_to_si(cluster.mass.sum(), 1.|units.Myr)

    timer_framework = np.zeros((len(timescales)))
    timer_codes = np.zeros((len(timescales), 2))

    R_com = np.zeros((len(timescales), 3)) | units.pc

    for i in range(len(timescales)):

        print (timescales[i].value_in(units.kyr), flush=True)

        system, gravity_cluster, gravity_galaxy = setup_bridged_gravity(
            ph4, timescales[i], converter)
        gravity_cluster.parameters.force_sync = True
        gravity_cluster.particles.add_particles(cluster.copy())

        system.record_runtime = True

        system.evolve_model(end_time)

        clusters_final = gravity_cluster.particles.copy()

        print (system.runtime_framework, system.runtime_codes)
        print ("Relative errors in model time:", 
            abs(([ code.model_time.value_in(units.Myr) for code in \
            system.codes if hasattr(code, 'model_time') ]|units.Myr) - system.model_time)/system.model_time)

        timer_framework[i] = system.runtime_framework
        timer_codes[i] = system.runtime_codes

        R_com[i] = clusters_final.center_of_mass()

    np.savetxt('data/R_com_bridge_constant.txt',
        np.array([timescales.value_in(units.kyr), 
            R_com[:,0].value_in(units.kpc), R_com[:,1].value_in(units.kpc),
            R_com[:,2].value_in(units.kpc)]).T)

    np.savetxt('data/timer_bridge_constant.txt',
        np.array([timescales.value_in(units.kyr), 
            timer_framework, timer_codes[:,0], timer_codes[:,1]]).T)


def run_convergence_bridge_adaptive (cluster, eta, end_time):

    converter = nbody_system.nbody_to_si(cluster.mass.sum(), 1.|units.Myr)

    timer_framework = np.zeros((len(eta)))
    timer_codes = np.zeros((len(eta), 2))

    R_com = np.zeros((len(eta), 3)) | units.pc

    for i in range(len(eta)):

        print (eta[i], flush=True)

        system, gravity_cluster, gravity_galaxy = setup_bridged_gravity_adaptive(
            ph4, eta[i], converter)
        gravity_cluster.parameters.force_sync = True
        gravity_cluster.particles.add_particles(cluster.copy())

        system.record_runtime = True

        system.evolve_model(end_time)

        clusters_final = gravity_cluster.particles.copy()

        print (system.runtime_framework, system.runtime_codes)
        print ("Relative errors in model time:", 
            abs(([ code.model_time.value_in(units.Myr) for code in \
            system.codes if hasattr(code, 'model_time') ]|units.Myr) - system.model_time)/system.model_time)

        timer_framework[i] = system.runtime_framework
        timer_codes[i] = system.runtime_codes

        R_com[i] = clusters_final.center_of_mass()

    np.savetxt('data/R_com_bridge_adaptive.txt',
        np.array([eta, 
            R_com[:,0].value_in(units.kpc), R_com[:,1].value_in(units.kpc),
            R_com[:,2].value_in(units.kpc)]).T)

    np.savetxt('data/timer_bridge_adaptive.txt',
        np.array([eta, 
            timer_framework, timer_codes[:,0], timer_codes[:,1]]).T)


def plot_convergence_bridge_constant ():

    fig = plt.figure(figsize=(6.4, 9.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale [kyr]')
    ax1.set_ylabel('Centre of mass error [kpc]')

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    ax1.text(3e1, 1e-4, 'dR~$\\tau^2$', rotation=45)

    ax1.set_xlim(3e0, 3e4)
    ax1.set_ylim(3e-6, 3e0)


    ax2.set_xlabel('Coupling timescale [kyr]')
    ax2.set_ylabel('Run-time [s]')

    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlim(3e0, 3e4)
    ax2.set_ylim(1e-1, 3e2)

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Cluster gravity')


    timescales, timer_framework, timer_cluster, timer_gravity = np.loadtxt(
        'data/timer_bridge_constant.txt', unpack=True)

    _, x_com, y_com, z_com = np.loadtxt(
        'data/R_com_bridge_constant.txt', unpack=True)

    dR_com = np.abs((x_com[1:]**2. + y_com[1:]**2. + z_com[1:]**2.) - (x_com[:-1]**2. + y_com[:-1]**2. + z_com[:-1]**2.))
    t_mid = (timescales[1:]*timescales[:-1])**0.5

    ax1.errorbar(t_mid, dR_com,
        xerr=(np.abs(t_mid-timescales[:-1]), np.abs(timescales[1:]-t_mid)),
        capsize=3., fmt='C0,:')

    ax1.plot(t_mid, dR_com[len(t_mid)//2] * (t_mid/t_mid[len(t_mid)//2])**2,
        c='k', label='$dR\\sim\\tau^{a}$', linestyle='--')


    timer = timer_framework + timer_cluster + timer_gravity


    ax2.plot(timescales, timer_framework, 
        linestyle='-', c='C0')
    ax2.plot(timescales, timer_cluster, 
        linestyle='--', c='C0')


    ax2.legend(frameon=False)

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('figures/bridge_constant_convergence.png', bbox_inches='tight')
    fig.savefig('figures/bridge_constant_convergence.pdf', bbox_inches='tight')


def plot_convergence_bridge_adaptive ():

    fig = plt.figure(figsize=(6.4, 9.6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('Coupling timescale parameter $\\eta$')
    ax1.set_ylabel('Centre of mass error [kpc]')

    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    ax1.set_xlim(3e-4, 3e0)
    ax1.set_ylim(1e-5, 1e2)


    ax2.set_xlabel('Coupling timescale parameter $\\eta$')
    ax2.set_ylabel('Run-time [s]')

    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlim(3e-4, 3e0)
    ax2.set_ylim(3e-2, 3e2)

    ax2.plot(0., 0., linestyle='-', c='k', label='Framework')
    ax2.plot(0., 0., linestyle='--', c='k', label='Cluster gravity')


    eta, timer_framework, timer_cluster, timer_gravity = np.loadtxt(
        'data/timer_bridge_adaptive.txt', unpack=True)

    _, x_com, y_com, z_com = np.loadtxt(
        'data/R_com_bridge_adaptive.txt', unpack=True)

    dR_com = np.abs((x_com[1:]**2. + y_com[1:]**2. + z_com[1:]**2.) - (x_com[:-1]**2. + y_com[:-1]**2. + z_com[:-1]**2.))
    eta_mid = (eta[1:]*eta[:-1])**0.5

    ax1.errorbar(eta_mid, dR_com,
        xerr=(np.abs(eta_mid-eta[:-1]), np.abs(eta[1:]-eta_mid)),
        capsize=3., fmt='C0,:')

    ax1.plot(eta_mid, dR_com[len(eta_mid)//2] * (eta_mid/eta_mid[len(eta_mid)//2])**2,
        c='k', label='$dR\\sim\\eta^{a}$', ls='--')
    ax1.text(3e-3, 3e-4, 'dR~$\\eta^2$', rotation=45)


    timer = timer_framework + timer_cluster + timer_gravity


    ax2.plot(eta, timer_framework, 
        linestyle='-', c='C0')
    ax2.plot(eta, timer_cluster, 
        linestyle='--', c='C0')


    ax2.legend(frameon=False)

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('figures/bridge_adaptive_convergence.png', bbox_inches='tight')
    fig.savefig('figures/bridge_adaptive_convergence.pdf', bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eccentricity', dest='ecc', type=float, default=0.)
    args = parser.parse_args()

    np.random.seed(49023723)

    N_cluster = 100
    radius = 10. | units.pc

    timescales = [0.01, 0.03, 0.1, 0.3, 1., 3., 10.] | units.Myr
    eta = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.])

    cluster = generate_cluster(N_cluster, radius)
    cluster.radius = radius

    cluster.x += 8. | units.kpc
    cluster_com = cluster.center_of_mass()

    ax, ay, az = MWpotentialBovy2015().get_gravity_at_point(0.|units.pc, cluster_com[0], cluster_com[1], cluster_com[2])
    a2 = ax*ax + ay*ay + az*az
    v2_kep = cluster_com.length()*a2**0.5
    end_time = (4.*np.pi**2*cluster_com.length()*a2**-0.5)**0.5

    if args.ecc == 0.:
        cluster.vy += ((1.-args.ecc)/(1.+args.ecc)*v2_kep)**0.5
        #run_convergence_bridge_constant(cluster, timescales, end_time)
        plot_convergence_bridge_constant()

    else:
        cluster.vy += ((1.-args.ecc)/(1.+args.ecc)*v2_kep)**0.5
        #run_convergence_bridge_adaptive(cluster, eta, end_time)
        plot_convergence_bridge_adaptive()

    plt.show()
