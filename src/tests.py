import numpy as np
import matplotlib.pyplot as plt

from venice import Venice

from amuse.units import units, constants, nbody_system
from amuse.datamodel import Particles
from amuse.ic.plummer import new_plummer_model
from amuse.ic.brokenimf import MultiplePartIMF
from amuse.community.seba.interface import SeBa
from amuse.community.sse.interface import SSE
from amuse.community.ph4.interface import ph4

import time


kroupa = MultiplePartIMF(
    mass_boundaries=[0.08, 0.5, 100.]|units.MSun,
    alphas=[-1.3, -2.3])


def make_cluster (N, R):

    mass = kroupa.next_mass(N)

    cluster = new_plummer_model(N, nbody_system.nbody_to_si(mass.sum(), R))

    cluster.mass = mass

    return cluster


def make_gravity_stellar (converter, timestep, verbose=False):

    stellar = SeBa()


    gravity = ph4(converter)
    gravity.parameters.force_sync = True


    system = Venice()
    system.verbose = verbose

    system.add_code(stellar)
    system.add_code(gravity)

    system.add_channel(0, 1, from_attributes=['mass'], to_attributes=['mass'])
    system.timestep_matrix[0,1] = timestep


    return stellar, gravity, system


def test_linear_vs_interlaced (N, R, timesteps, end_time):

    cluster_l = make_cluster(N, R)
    cluster_i = cluster_l.copy()

    converter = nbody_system.nbody_to_si(cluster_l.mass.sum(), R)


    fig = plt.figure()
    ax = fig.add_subplot(111)


    for i in range(len(timesteps)):

        stellar_l, gravity_l, system_l = make_gravity_stellar(converter, 
            timesteps[i])

        stellar_l.particles.add_particles(cluster_l)
        gravity_l.particles.add_particles(cluster_l)
        channel_l = gravity_l.particles.new_channel_to(cluster_l)


        stellar_i, gravity_i, system_i = make_gravity_stellar(converter,
            timesteps[i])
        system_i.interlaced_drift = True

        stellar_i.particles.add_particles(cluster_i)
        gravity_i.particles.add_particles(cluster_i)
        channel_i = gravity_i.particles.new_channel_to(cluster_i)


        start = time.time()
        system_l.evolve_model(end_time)
        end = time.time()
        channel_l.copy()
        print (i, 'linear', end-start)

        start = time.time()
        system_i.evolve_model(end_time)
        end = time.time()
        channel_i.copy()
        print (i, 'interlaced', end-start)


        dr = (cluster_l.position - cluster_i.position).lengths().value_in(units.pc)
        hist_r, bins_r = np.histogram( np.log10(dr[dr > 0. ]), bins=30)

        dv = (cluster_l.velocity - cluster_i.velocity).lengths().value_in(units.kms)
        hist_v, bins_v = np.histogram( np.log10(dv[ dv > 0. ]), bins=30)


        ax.plot((bins_r[1:] + bins_r[:-1])/2., hist_r, 
            drawstyle='steps-mid', c='C'+str(i),
            label=timesteps[i].value_in(units.kyr))
        ax.plot((bins_v[1:] + bins_v[:-1])/2., hist_v, 
            drawstyle='steps-mid', c='C'+str(i), linestyle='--')

    ax.set_xlabel('Error [pc, km/s]')
    ax.set_ylabel('N')

    ax.legend()


if __name__ == '__main__':

    N = 1000
    R = 3. | units.pc

    timesteps = [1., 0.3, 0.1] | units.kyr
    end_time = 10. | units.kyr

    test_linear_vs_interlaced (N, R, timesteps, end_time)

    plt.show()

