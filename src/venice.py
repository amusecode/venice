import numpy as np

import time

from amuse.units import units

from symmetric_matrix import SymmetricMatrix


def find_connected_components (code_ids, dt, timestep_matrix):
    '''
    Find connected components among the timestep matrix of a set of codes

    code_ids: the indices of the codes to find connected components for
        (int array length (N))
    dt: pivot timestep of the connected component (scalar, units of time)
    timestep_matrix: matrix containing the target bridge timesteps 
        (array length (M,M), units of time, all entries in code_ids < M)
    '''

    N = len(code_ids)
    component_ids = -np.ones(N, dtype=int)

    counter = 0

    for x in range(N):

        if component_ids[x] < 0:

            component_ids[x] = counter

            stack = [x]

            while len(stack):

                i = stack.pop()
                I = code_ids[i]

                for j in range(N):

                    J = code_ids[j]

                    if component_ids[j] < 0 and timestep_matrix[I,J] < dt \
                            and timestep_matrix[I,J] > 0. | units.s:

                        component_ids[j] = counter
                        stack.append(j)

            counter += 1

    return component_ids


class NodeCodes:
    '''
    Node of a Venice connected component tree
    '''

    def __init__ (self, code_ids, dt, timestep_matrix):
        '''
        code_ids: the indices of the codes in this node (int array length (N))
        dt: pivot timestep of the node (scalar, units of time)
        timestep_matrix: matrix containing the target bridge timesteps 
            (array length (M,M), units of time, all entries in code_ids < M)

        Note that timestep_matrix could be reduced to (N,N) as only data for
        code_ids present is necessary; however, I am lazy.
        '''

        self.N_codes = len(code_ids)
        self.code_ids = code_ids

        self.dt = dt

        self.subdivide_node(timestep_matrix)


    def subdivide_node (self, timestep_matrix):
        '''
        Subdivide this node in connected components

        timestep_matrix: coupling timesteps of Venice system
        '''

        self.children = []

        if self.N_codes > 1:

            component_ids = find_connected_components(self.code_ids, self.dt,
                timestep_matrix)

            self.N_children = np.max(component_ids)+1

            for i in range(self.N_children):

                mask = component_ids == i

                self.children.append( NodeCodes(self.code_ids[ mask ], self.dt/2.,
                    timestep_matrix) )

        else:

            self.N_children = 0


    def print_node (self):

        print ("Node at dt = {a} yr; codes ".format(a=self.dt.value_in(units.yr)),
            self.code_ids)
        for child in self.children:
            child.print_node()


class Venice:

    def __init__ (self):

        self.codes = []

        self.timestep_matrix = None

        self.kick = None
        self.update_timestep = None
        self.sync_data = None
        self._channels = None

        self.save_data = []

        self.model_time = 0. | units.s

        self.rest_order = 2
        self.cc_order = 2

        self.verbose = False

        self.io_scheme = 0
        self.filepath = './'
        self._chk_counters = []
        self._plt_counters = []
        self._dbg_counters = []

        self.record_runtime = False
        self.runtime_total = 0.
        self.runtime_framework = 0.
        self.runtime_codes = []


    '''
    How to build Venice after initializing the class:

    Fundamentals
    - Add codes
        - using add_code(code)
    - Define coupling times
        - timestep_matrix[i,j] contains the maximum coupling timescale between codes
            i and j
        - the actual coupling timescale will be between 1 and 1/2 this value
        - if not defined (or set to <= 0), the coupling timescale is the total
            evolution time
        - timestep_matrix is automatically symmetric
    - Define channels
        - add_channel(i,j) adds a channel from a dataset of code i to one of code j
        - the default dataset is particles, but channels can be built between any
            Particles or Grid datasets
        - by default all properties are copied, but these can be set to any subset
        - add_iterable_channels and add_dynamic_iterable_channels function
            similarly, but for datasets that are iterated over (e.g. via the
            itergrids function in AMR codes). Note that add_iterable_channels adds
            a static set of channels, so is unsuited for codes with e.g. a dynamic
            set of grids. add_dynamic_iterable_grids makes channels when called,
            this is slower but can be used if the set of grids changes. 

    Bells & Whistles
    - Define kick functions
        - kick[i][j] describes an effect of code i on code j, which doesn't change
            code i
        - kick functions must be of the form kick_function(code i, code j,
            timestep)
        - each entry is a single function, multiple kick interactions must be
            aggregated to a single function
    - Define data synchronizers
        - sync_data[i][j] updates the structure of the dataset of code j with any
            changes made by code i
        - sync_data functions must be of the form sync_data(code i, code j)
        - this concerns e.g. added/removed particles and (de)refined grids
    - Define timestep updaters
        - update_timestep[i][j] computes the coupling timestep of codes i and j
          during evolution, must return a scalar with units of time
        - update_timestep functions must be of the form update_timestep(code i,
          code j, previous time step)
        - the tree will be updated to accomodate the new timesteps

    IO Strategy
    - Venice handles IO through user-defined save_data functions. These are of the
      form save_data(code i, filename). Filename will be defined by Venice, and
      includes the path to the output directory. This string can be formatted 
      through the 'code' and 'set' variable to differentiate between different 
      codes and data sets within the code, e.g.:
        write_set_to_file(code.gas_particles, filename.format(code='hydro', set=
            'gas_particles')
        write_set_to_file(code.dm_particles, filename.format(code='hydro', set=
            'dm_particles')
    - There are four (cumulative) levels of automated IO:
        - 0: None
            - No files are written by Venice, although codes can write internally,
              and the final state is available through Venice itself.
        - 1: Checkpoints
            - Data is saved at the smallest time resolution where the entire system
              is synchronized. In the connected component tree structure, this is
              at the end of evolution of the highest node with more than one child.
              This allows restarts.
        - 2: Plot
            - Data is saved at the end of evolution of every node with more than one
              child. This prevents duplicates in tightly coupled branches.
        - 3: Debug
            - Data is saved at the end of every evolve call.
    '''


    def evolve_model (self, end_time):
        '''
        Evolve the Venice system to an end time

        end_time: time to evolve to (scalar, units of time)
        '''

        if self.record_runtime:
            start = time.time()

        dt = end_time - self.model_time

        self.root = NodeCodes(np.arange(len(self.codes)), dt, self.timestep_matrix)

        # Determine if dynamical timesteps are used (and cc tree needs recomputing)
        self._dynamic_timesteps = False
        for i in range(len(self.codes)):
          for j in range(len(self.codes)):
              if self.update_timestep[i][j] is not None:
                  self._dynamic_timesteps = True
                  break
          if self._dynamic_timesteps:
              break

        self._evolve_cc(self.root, dt, False, 
            len(self.root.children) > 1, len(self.root.children) > 1)

        self.model_time = end_time

        if self.record_runtime:
            end = time.time()
            self.runtime_total += end - start
            self.runtime_framework = self.runtime_total - np.sum(self.runtime_codes)

        if self.verbose:
            print ("Relative errors in model time:", 
                abs(([ code.model_time.value_in(units.Myr) for code in \
                self.codes ]|units.Myr) - self.model_time)/self.model_time)


    def _evolve_cc (self, node, dt, reorganize, save_chk, save_plt):
        '''
        Evolve a node in the cc tree for a timestep

        node: cc tree node id to evolve (int)
        dt: timestep to evolve for (scalar, units of time)
        reorganize: flag to recompute tree structure from this node down (bool)
        save_chk: flag to save checkpoint file (bool)
        save_plt: flag to save plot file (bool)
        '''

        # Only update tree if:
        #   a) this is after the second subcall
        #   b) there are dynamic timesteps (otherwise the tree is static)
        if reorganize and self._dynamic_timesteps:
            # UPDATE TIMESTEPS
            for code_id1 in node.code_ids:
              for code_id2 in node.code_ids:
                  if self.update_timestep[code_id1][code_id2] is not None:
                      self.timestep_matrix[code_id1, code_id2] = \
                          self.update_timestep[code_id1][code_id2](
                              self.codes[code_id1], self.codes[code_id2], dt)

            # UPDATE CC TREE
            node.subdivide_node(self.timestep_matrix)


        CC = [ child for child in node.children if child.N_codes > 1 ]
        R  = [ child.code_ids[0] for child in node.children if child.N_codes == 1 \
            and hasattr(self.codes[child.code_ids[0]], 'evolve_model') ]


        # RECURSIVE EVOLUTION
        if len(CC):
            if self.cc_order == 1:
                self._evolve_split_cc_1st_order(CC, dt/2., False, save_chk,save_plt)
            elif self.cc_order == 2:
                self._evolve_split_cc_2nd_order(CC, dt/2., False, save_chk,save_plt)
            else:
                print ("[CC] Order {a} CC splitting is not implemented!".format(
                    a=self.cc_order))
                return


        # KICK
        for child1 in node.children:
          for child2 in node.children:
              if child1 != child2:
                  for code_id1 in child1.code_ids:
                    for code_id2 in child2.code_ids:
                        if self.kick[code_id1][code_id2] is not None:
                            if self.verbose:
                                print ("Kicking from code {a} to code {b} for {c} kyr".format(
                                    a=code_id1, b=code_id2, 
                                    c=dt.value_in(units.kyr)/2.))
                            self.kick[code_id1][code_id2](self.codes[code_id1],
                                self.codes[code_id2], dt/2.)
                            self._sync_code_to_codes(code_id2, node.code_ids)
                            self._copy_code_to_codes(code_id2, node.code_ids)


        # DRIFT
        if len(R):
            if self.rest_order == 1:
                self._evolve_codes_1st_order(R, dt)
            elif self.rest_order == 2:
                self._evolve_codes_2nd_order(R, dt)
            else:
                print ("[CC] Order {a} R splitting is not implemented!".format(
                    a=self.cc_order))
                return


        # KICK
        for child1 in node.children:
          for child2 in node.children:
              if child1 != child2:
                  for code_id1 in child1.code_ids:
                    for code_id2 in child2.code_ids:
                        if self.kick[code_id1][code_id2] is not None:
                            if self.verbose:
                                print ("Kicking from code {a} to code {b} for {c} kyr".format(
                                    a=code_id1, b=code_id2, 
                                    c=dt.value_in(units.kyr)/2.))
                            self.kick[code_id1][code_id2](self.codes[code_id1],
                                self.codes[code_id2], dt/2.)
                            self._sync_code_to_codes(code_id2, node.code_ids)
                            self._copy_code_to_codes(code_id2, node.code_ids)


        # RECURSIVE EVOLUTION
        if len(CC):
            if self.cc_order == 1:
                self._evolve_split_cc_1st_order(CC, dt/2., True, save_chk, save_plt)
            elif self.cc_order == 2:
                self._evolve_split_cc_2nd_order(CC, dt/2., True, save_chk, save_plt)
            else:
                print ("[CC] Order {a} CC splitting is not implemented!".format(
                    a=self.cc_order))
                return


        # SAVE CHECKPOINT FILE
        if self.io_scheme > 0 and save_chk:
            for code_id in node.code_ids:
                if self.save_data[code_id] is not None:
                    self.save_state[code_id](self.codes[code_id], 
                        self.filepath + '/chk_i{i:06}'.format(
                            i=self._chk_counters[code_id]) + \
                        '_{code}_{set}.venice')
                    self._chk_counters[code_id] += 1

        # SAVE PLOT FILE
        if self.io_scheme > 1 and save_plt:
            for code_id in node.code_ids:
                if self.save_data[code_id] is not None:
                    self.save_state[code_id](self.codes[code_id], 
                        self.filepath + '/plt_i{i:06}'.format(
                            i=self._plt_counters[code_id]) + \
                        '_{code}_{set}.venice')
                    self._plt_counters[code_id] += 1


    def _evolve_codes_1st_order (self, code_ids, dt):
        '''
        Evolve multiple codes with linear coupling for a timestep
        Codes are evolved one by one for the full timestep, with coupling in between
        Error scales with O(t)

        code_ids: Venice code ids to evolve (list of ints)
        dt: timestep to evolve (scalar, units of time)
        '''

        for code_id in code_ids:

            if self.verbose:
                print ("[1] Evolving code {a} for {b} kyr".format(
                    a=code_id, b=dt.value_in(units.kyr)))

            if self.record_runtime:
                start = time.time()

            self.codes[code_id].evolve_model( self.codes[code_id].model_time + dt )

            if self.record_runtime:
                end = time.time()
                self.runtime_codes[code_id] += end - start

            if self.io_scheme == 3 and self.save_data[code_id] is not None:
                self.save_data[code_id](self.codes[code_id], 
                    self.filepath + '/dbg_i{i:06}'.format(
                        i=self._dbg_counters[code_id]) + \
                    '_{code}_{set}.venice')
                self._dbg_counters[code_id] += 1

            self._sync_code_to_codes(code_id, code_ids)
            self._copy_code_to_codes(code_id, code_ids)


    def _evolve_codes_2nd_order (self, code_ids, dt):
        '''
        Evolve multiple codes with interlaced coupling for a timestep
        Codes are evolved using the general interlaced scheme, 
        with coupling in between
        Error scales with O(t^2)

        code_ids: Venice code ids to evolve (list of ints)
        dt: timestep to evolve (scalar, units of time)
        '''

        for i in range(len(code_ids)-1):
            if self.verbose:
                print ("[2] Evolving code {a} for {b} kyr".format(
                    a=code_ids[i], b=dt.value_in(units.kyr)/2.))

            if self.record_runtime:
                start = time.time()

            self.codes[code_ids[i]].evolve_model( 
                self.codes[code_ids[i]].model_time + dt/2. )

            if self.record_runtime:
                end = time.time()
                self.runtime_codes[code_ids[i]] += end - start

            if self.io_scheme == 3 and self.save_data[code_ids[i]] is not None:
                self.save_data[code_ids[i]](self.codes[code_ids[i]], 
                    self.filepath + '/dbg_i{a:06}'.format(
                        a=self._dbg_counters[code_ids[i]]) + \
                    '_{code}_{set}.venice')
                self._dbg_counters[code_ids[i]] += 1

            self._sync_code_to_codes(code_ids[i], code_ids)
            self._copy_code_to_codes(code_ids[i], code_ids)


        if self.verbose:
            print ("[2] Evolving code {a} for {b} kyr".format(
                a=code_ids[-1], b=dt.value_in(units.kyr)))

        if self.record_runtime:
            start = time.time()

        self.codes[code_ids[-1]].evolve_model( 
            self.codes[code_ids[-1]].model_time + dt )

        if self.record_runtime:
            end = time.time()
            self.runtime_codes[code_ids[-1]] += end - start

        if self.io_scheme == 3 and self.save_data[code_ids[-1]] is not None:
            self.save_data[code_ids[-1]](self.codes[code_ids[-1]], 
                self.filepath + '/dbg_i{a:06}'.format(
                    a=self._dbg_counters[code_ids[-1]]) + \
                '_{code}_{set}.venice')
            self._dbg_counters[code_ids[-1]] += 1

            self._sync_code_to_codes(code_ids[-1], code_ids)
            self._copy_code_to_codes(code_ids[-1], code_ids)


        for i in range(len(code_ids)-1)[::-1]:
            if self.verbose:
                print ("[2] Evolving code {a} for {b} kyr".format(
                    a=code_ids[i], b=dt.value_in(units.kyr)/2.))

            if self.record_runtime:
                start = time.time()

            self.codes[code_ids[i]].evolve_model( 
                self.codes[code_ids[i]].model_time + dt/2. )

            if self.record_runtime:
                end = time.time()
                self.runtime_codes[code_ids[i]] += end - start

            if self.io_scheme == 3 and self.save_data[code_ids[i]] is not None:
                self.save_data[code_ids[i]](self.codes[code_ids[i]], 
                    self.filepath + '/dbg_i{a:06}'.format(
                        a=self._dbg_counters[code_ids[i]]) + \
                    '_{code}_{set}.venice')
                self._dbg_counters[code_ids[i]] += 1

            self._sync_code_to_codes(code_ids[i], code_ids)
            self._copy_code_to_codes(code_ids[i], code_ids)


    def _evolve_split_cc_1st_order (self, nodes, dt, reorganize, 
            save_chk, save_plt):
        '''
        Evolve a connected component with linear coupling for a timestep
        Connected components are evolved one by one for the full timestep, with 
        coupling in between
        Error scales with O(t)

        nodes: list of connected component nodes, each with >1 code 
            (list of NodeCodes object)
        dt: timestep to evolve (scalar, units of time)
        reorganize: reorganize the CC tree in the next call
        save_chk: save a checkpoint file in the next call
        save_plt: save a plot file in the next call
        '''

        for i in range(len(nodes)):
            self._evolve_cc(nodes[i], dt, reorganize, 
                # Save chk if in topmost, >1 child node, node of tree
                (len(nodes[i].children)>1) * \
                (nodes[i].N_codes==len(self.root.code_ids)),
                # Save plt if in >1 child node, node of tree
                 len(nodes[i].children)>1)
            for code_id in nodes[i].code_ids:
                for j in range(len(nodes)):
                    if i != j:
                        self._sync_code_to_codes(code_id, nodes[j].code_ids)
                        self._copy_code_to_codes(code_id, nodes[j].code_ids)


    def _evolve_split_cc_2nd_order (self, nodes, dt, reorganize,
            save_chk, save_plt):
        '''
        Evolve a connected component with interlaced coupling for a timestep
        Connected components are evolved using the general interlaced scheme, 
        with coupling in between
        Error scales with O(t^2)

        nodes: list of connected component nodes, each with >1 code 
            (list of NodeCodes object)
        dt: timestep to evolve (scalar, units of time)
        reorganize: reorganize the CC tree in the next call
        save_chk: save a checkpoint file in the next call
        save_plt: save a plot file in the next call
        '''

        for i in range(len(nodes)-1):
            self._evolve_cc(nodes[i], dt/2., False,
                # Save chk if in topmost, >1 child node, node of tree
                (len(nodes[i].children)>1) * \
                (nodes[i].N_codes==len(self.root.code_ids)),
                # Save plt if in >1 child node, node of tree
                 len(nodes[i].children)>1)
            for code_id in nodes[i].code_ids:
                for j in range(len(nodes)):
                    if i != j:
                        self._sync_code_to_codes(code_id, nodes[j].code_ids)
                        self._copy_code_to_codes(code_id, nodes[j].code_ids)

        self._evolve_cc(nodes[-1], dt, False,
            # Save chk if in topmost, >1 child node, node of tree
            (len(nodes[-1].children)>1) * \
            (nodes[-1].N_codes==len(self.root.code_ids)),
            # Save plt if in >1 child node, node of tree
             len(nodes[-1].children)>1)
        for code_id in nodes[-1].code_ids:
            for j in range(len(nodes)-1):
                self._sync_code_to_codes(code_id, nodes[j].code_ids)
                self._copy_code_to_codes(code_id, nodes[j].code_ids)

        for i in range(len(nodes)-1)[::-1]:
            self._evolve_cc(nodes[i], dt/2., reorganize,
                # Save chk if in topmost, >1 child node, node of tree
                (len(nodes[i].children)>1) * \
                (nodes[i].N_codes==len(self.root.code_ids)),
                # Save plt if in >1 child node, node of tree
                 len(nodes[i].children)>1)
            for code_id in nodes[i].code_ids:
                for j in range(len(nodes)):
                    if i != j:
                        self._sync_code_to_codes(code_id, nodes[j].code_ids)
                        self._copy_code_to_codes(code_id, nodes[j].code_ids)


    def _copy_code_to_codes (self, from_code_id, to_code_ids):
        '''
        Copy data from one code to a list of codes

        from_code_id: Venice code id to copy from (int)
        to_code_ids: Venice code ids to copy to (list of ints)
        '''

        for to_code_id in to_code_ids:
            for channel in self._channels[from_code_id][to_code_id]:
                if self.verbose:
                    print ("Copying from code {a} to code {b}".format(
                        a=from_code_id, b=to_code_id))
                channel.copy()


    def _sync_code_to_codes (self, from_code_id, to_code_ids):
        '''
        Synchronize data from one code to a list of codes

        from_code_id: Venice code id to copy from (int)
        to_code_ids: Venice code ids to copy to (list of ints)
        '''

        for to_code_id in to_code_ids:
            if self.sync_data[from_code_id][to_code_id] is not None:
                self.sync_data[from_code_id][to_code_id](
                    self.codes[from_code_id], self.codes[to_code_id])


    def add_channel (self, from_code_id, to_code_id, from_set_name='particles', 
            to_set_name='particles', from_attributes=None, to_attributes=None):
        '''
        Add a channel from one code's dataset to another code's dataset

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_set_name: dataset to copy from (string)
        to_set_name: dataset to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        self._channels[from_code_id][to_code_id].append(
            getattr(self.codes[from_code_id], from_set_name).new_channel_to(
            getattr(self.codes[to_code_id], to_set_name),
            attributes=from_attributes, target_names=to_attributes)
        )


    def add_iterable_channels (self, from_code_id, to_code_id,
            from_iterator='itergrids', to_iterator='itergrids',
            from_attributes=None, to_attributes=None):
        '''
        Add channels from one code's iterable datasets to another code's iterable 
        datasets
        Use only if the set of grids doesn't vary, use add_dynamic_iterable_channels
        instead, but that method is slower!

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_iterator: dataset iterator to copy from (string)
        to_iterator: dataset iterator to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        for from_grid, to_grid in zip(
                getattr(self.codes[from_code_id], from_iterator)(), 
                getattr(self.codes[to_code_id], to_iterator)()):

            self._channels[from_code_id][to_code_id].append(
                from_grid.new_channel_to(to_grid, 
                attributes=from_attributes, target_names=to_attributes)
            )


    def add_dynamic_iterable_channels (self, from_code_id, to_code_id,
            from_iterator='itergrids', to_iterator='itergrids',
            from_attributes=None, to_attributes=None):
        '''
        Add a dynamic channel generator and copier from one code's iterable
        datasets to another code's iterable datasets
        Use over add_iterable_channels when the set of iterable grids can vary,
        but this method is slower

        from_code: Venice code id of code to copy from (int)
        to_code: Venice code id of code to copy to (int)
        from_iterator: dataset iterator to copy from (string)
        to_iterator: dataset iterator to copy to (string)
        from_attributes: list of attributes to copy from code (list of strings)
        to_attributes: list of attributes to copy to code (list of strings)
        '''

        self._channels[from_code_id][to_code_id].append(
            DynamicIterableChannel(
                getattr(self.codes[from_code_id], from_iterator),
                getattr(self.codes[to_code_id], to_iterator),
                from_attributes=from_attributes, to_attributes=to_attributes)
            )


    def add_code (self, code):
        '''
        Add a code to Venice. See comments under __init__ for other
        initializations required to couple this code to the others.

        code: code to add to Venice
        '''

        N_codes = len(self.codes)


        self.codes.append(code)

        self.save_data.append(None)
        self._chk_counters.append(1)
        self._plt_counters.append(1)
        self._dbg_counters.append(1)

        self.runtime_codes.append(0)


        if N_codes > 0:
            for i in range(N_codes):
                self.kick[i].append(None)
                self.update_timestep[i].append(None)
                self.sync_data[i].append(None)
                self._channels[i].append([])

            self.kick.append([ None for _ in range(N_codes+1) ])
            self.update_timestep.append([ None for _ in range(N_codes+1) ])
            self.sync_data.append([ None for _ in range(N_codes+1) ])
            self._channels.append([ [] for _ in range(N_codes+1) ])


            new_timestep_matrix = SymmetricMatrix(N_codes+1, units.s)
            for i in range(N_codes):
                for j in range(i+1,N_codes):
                    new_timestep_matrix[i,j] = self.timestep_matrix[i,j]
            self.timestep_matrix = new_timestep_matrix

        else:
            self.kick = [[None]]
            self.update_timestep = [[None]]
            self.sync_data = [[None]]
            self._channels = [[[]]]

            self.timestep_matrix = SymmetricMatrix(1, units.s)


    def stop (self):
        '''
        Stop every code that has a stop function
        '''

        for code in self.codes:
            if hasattr(code, 'stop'):
                code.stop()


def dynamic_kick (kicker, kickee, dt):
    '''
    Kick function for classic dynamical kick. Lets a 'kicker' code provide velocity
    kicks on a 'kickee' code for a certain timestep.

    kicker: class that provides dynamical kicks. Must have a get_gravity_at_point
        function defined (class)
    kickee: class with particles that feel the dynamical kicks (class)
    dt: timestep to kick for (scalar, units of time)
    '''

    kickee_particles = kickee.particles.copy(
        filter_attributes=lambda p, attribute_name: \
            attribute_name in ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    ax, ay, az = kicker.get_gravity_at_point(0. | kicker.particles.position.unit,
        kickee_particles.x, kickee_particles.y, kickee_particles.z)

    kickee_particles.vx += ax * dt
    kickee_particles.vy += ay * dt
    kickee_particles.vz += az * dt

    channel = kickee_particles.new_channel_to(kickee.particles)
    channel.copy_attributes(['vx', 'vy', 'vz'])


class DynamicKick:

    def __init__ (self, radius_is_eps=False, h_smooth_is_eps=False,
            zero_smoothing=False):

        self.radius_is_eps = radius_is_eps
        self.h_smooth_is_eps = h_smooth_is_eps
        self.zero_smoothing = zero_smoothing


    def __call__ (self, kicker, kickee, dt):
        '''
        Kick function for classic dynamical kick. Lets a 'kicker' code provide 
        velocity kicks on a 'kickee' code for a certain timestep.

        kicker: class that provides dynamical kicks. Must have a 
        get_gravity_at_point function defined (class)
        kickee: class with particles that feel the dynamical kicks (class)
        dt: timestep to kick for (scalar, units of time)
        '''

        ax, ay, az = kicker.get_gravity_at_point(
            self._softening_lengths(kickee),
            kickee.particles.x, kickee.particles.y, kickee.particles.z)

        kickee.particles.vx += ax * dt
        kickee.particles.vy += ay * dt
        kickee.particles.vz += az * dt


    def _softening_lengths (self, code):
        if self.radius_is_eps:
            return code.particles.radius
        elif self.h_smooth_is_eps:
            return code.particles.h_smooth
        elif zero_smoothing:
            return 0.*code.particles.x
        elif hasattr(code, 'parameters') and hasattr(code.parameters, 
                'epsilon_squared'):
            return (code.parameters.epsilon_squared**0.5).as_vector_with_length(
                len(code.particles))
        else:
            return 0.*code.particles.x


class DynamicIterableChannel:
    '''
    Utility class for channels between codes that have a variable number of grids
    '''

    def __init__ (self, from_iterator, to_iterator,
            from_attributes=None, to_attributes=None):

        self.from_iterator = from_iterator
        self.to_iterator = to_iterator

        self.from_attributes = from_attributes
        self.to_attributes = to_attributes


    def copy (self):

        for from_grid, to_grid in zip(self.from_iterator(), self.to_iterator()):

            temp_channel = from_grid.new_channel_to(to_grid, 
                attributes=from_attributes, target_names=to_attributes)

            temp_channel.copy()
