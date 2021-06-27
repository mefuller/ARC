"""
An adapter for executing TS guess jobs based on heuristics

Todo:
    - eventually this module needs to be applied for all H's connected to the same heavy atom,
      summing up the rates appropriately
    - test H2O2 as the RH abstractor, see that both TS chiralities are attained
    - Add tests
    - add database and train from database, see https://github.com/ReactionMechanismGenerator/ARC/commit/081df5bf8e53987e9ff48eef481c17997f9cff2a, https://github.com/ReactionMechanismGenerator/ARC/commit/9a569ee80331494dcca26490fd66accc69697380, https://github.com/ReactionMechanismGenerator/ARC/commit/10d255467334f7821547e7dc5d98bb50bbfab7c0
"""

import datetime
import itertools
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from rmgpy.exceptions import ActionError
from rmgpy.reaction import Reaction

from arc.common import almost_equal_coords, colliding_atoms, get_logger, key_by_val
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import check_argument_consistency, ts_adapters_by_rmg_family
from arc.job.factory import register_job_adapter
from arc.plotter import save_geo
from arc.species.converter import compare_zmats, zmat_from_xyz, zmat_to_xyz
from arc.species.converter import xyz_from_data
from arc.species.species import ARCSpecies, TSGuess
from arc.species.zmat import get_parameter_from_atom_indices, is_angle_linear, up_param

if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import KineticsFamily
    from rmgpy.molecule.molecule import Molecule
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()


class HeuristicsAdapter(JobAdapter):
    """
    A class for executing TS guess jobs based on heuristics.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        conformer (int, optional): Conformer number if optimizing conformers.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime or str, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_status (list, optional): The job's server and ESS statuses.
        level (Level, optionnal): The level of theory to use.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        times_rerun (int, optional): Number of times this job was re-run with the same arguments (no trsh methods).
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsion(s).
        tsg (int, optional): TSGuess number if optimizing TS guesses.
        xyz (dict, optional): The 3D coordinates to use. If not give, species.get_xyz() will be used.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Union[List[str], str],
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 conformer: Optional[int] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedrals: Optional[List[float]] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional[Union['datetime.datetime', str]] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 level: Optional['Level'] = None,
                 max_job_time: Optional[float] = None,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 testing: bool = False,
                 times_rerun: int = 0,
                 torsions: Optional[List[List[int]]] = None,
                 tsg: Optional[int] = None,
                 xyz: Optional[dict] = None,
                 ):

        self.job_adapter = 'heuristics'
        self.execution_type = execution_type or 'incore'

        if reactions is None:
            raise ValueError('Cannot execute TS Heuristics without ARCReaction object(s).')

        self.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
        self.job_type = job_type
        self.project = project
        self.project_directory = project_directory
        if self.project_directory and not os.path.isdir(self.project_directory):
            os.makedirs(self.project_directory)
        self.args = args or dict()
        self.bath_gas = bath_gas
        self.checkfile = checkfile
        self.conformer = conformer
        self.constraints = constraints or list()
        self.cpu_cores = cpu_cores
        self.dihedrals = dihedrals
        self.ess_settings = ess_settings
        self.ess_trsh_methods = ess_trsh_methods or list()
        self.fine = fine
        self.initial_time = datetime.datetime.strptime(initial_time, '%Y-%m-%d %H:%M:%S') \
            if isinstance(initial_time, str) else initial_time
        self.irc_direction = irc_direction
        self.job_id = job_id
        self.job_memory_gb = job_memory_gb
        self.job_name = job_name
        self.job_num = job_num
        self.job_server_name = job_server_name
        self.job_status = job_status \
            or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
        self.level = level
        self.max_job_time = max_job_time
        self.reactions = [reactions] if reactions is not None and not isinstance(reactions, list) else reactions
        self.rotor_index = rotor_index
        self.server = server
        self.server_nodes = server_nodes or list()
        self.species = [species] if species is not None and not isinstance(species, list) else species
        self.testing = testing
        self.torsions = torsions
        self.tsg = tsg
        self.xyz = xyz
        self.times_rerun = times_rerun

        self.species_label = self.reactions[0].ts_species.label if self.reactions[0].ts_species is not None \
            else f'TS_{self.job_num}'  # The ts_species attribute should be initialized in a normal ARC run
        if len(self.reactions) > 1:
            self.species_label += f'_and_{len(self.reactions) - 1}_others'

        if self.job_num is None or self.job_name is None or self.job_server_name:
            self._set_job_number()

        self.args = dict()

        self.final_time = None
        self.run_time = None
        self.charge = None
        self.multiplicity = None
        self.is_ts = True
        self.scan_res = None
        self.set_file_paths()

        self.tasks = None
        self.iterate_by = list()
        self.number_of_processes = 0
        self.determine_job_array_parameters()  # Writes the local HDF5 file if needed.

        self.files_to_upload = list()
        self.files_to_download = list()
        self.set_files()  # Set the actual files (and write them if relevant).

        if job_num is None:
            # This checks job_num and not self.job_num on purpose.
            # If job_num was given, then don't save as initiated jobs, this is a restarted job.
            self._write_initiated_job_to_csv_file()

        check_argument_consistency(self)

    def write_input_file(self) -> None:
        """
        Write the input file to execute the job on the server.
        """
        pass

    def set_files(self) -> None:
        """
        Set files to be uploaded and downloaded. Writes the files if needed.
        Modifies the self.files_to_upload and self.files_to_download attributes.

        self.files_to_download is a list of remote paths.

        self.files_to_upload is a list of dictionaries, each with the following keys:
        ``'name'``, ``'source'``, ``'make_x'``, ``'local'``, and ``'remote'``.
        If ``'source'`` = ``'path'``, then the value in ``'local'`` is treated as a file path.
        Else if ``'source'`` = ``'input_files'``, then the value in ``'local'`` will be taken
        from the respective entry in inputs.py
        If ``'make_x'`` is ``True``, the file will be made executable.
        """
        pass

    def set_additional_file_paths(self) -> None:
        """
        Set additional file paths specific for the adapter.
        Called from set_file_paths() and extends it.
        """
        pass

    def set_input_file_memory(self) -> None:
        """
        Set the input_file_memory attribute.
        """
        pass

    def execute_incore(self):
        """
        Execute a job incore.
        """
        self._log_job_execution()
        self.initial_time = self.initial_time if self.initial_time else datetime.datetime.now()

        supported_families = [key for key, val in ts_adapters_by_rmg_family.items() if 'heuristics' in val]

        self.reactions = [self.reactions] if not isinstance(self.reactions, list) else self.reactions
        for rxn in self.reactions:
            family = rxn.family.label
            if family not in supported_families:
                logger.warning(f'The heuristics TS search adapter does not yet support the {family} reaction family.')
                continue
            if any(spc.get_xyz() is None for spc in rxn.r_species + rxn.p_species):
                logger.warning(f'The heuristics TS search adapter cannot process a reaction if 3D coordinates of '
                               f'some/all of its reactants/products are missing. Not processing {rxn}.')
                continue

            if rxn.ts_species is None:
                # Mainly used for testing, in an ARC run the TS species should already exist.
                rxn.ts_species = ARCSpecies(label='TS',
                                            is_ts=True,
                                            charge=rxn.charge,
                                            multiplicity=rxn.multiplicity,
                                            )
            rxn.arc_species_from_rmg_reaction()
            reactants, products = rxn.get_reactants_and_products(arc=True)
            reactant_mol_combinations = list(
                itertools.product(*list(reactant.mol_list for reactant in reactants)))
            product_mol_combinations = list(
                itertools.product(*list(product.mol_list for product in products)))
            reaction_list = list()
            for reactants in list(reactant_mol_combinations):
                for products in list(product_mol_combinations):
                    reaction = label_molecules(reactants=list(reactants),
                                               products=list(products),
                                               family=rxn.family,
                                               )
                    if reaction is not None:
                        reaction_list.append(reaction)

            xyzs = list()
            tsg = None
            if family == 'H_Abstraction':
                tsg = TSGuess(method=f'Heuristics')
                tsg.tic()
                xyzs = h_abstraction(arc_reaction=rxn,
                                     rmg_reactions=reaction_list,
                                     dihedral_increment=20,
                                     )  # todo: dihedral_increment should vary
                tsg.tok()

            for method_index, xyz in enumerate(xyzs):
                unique = True
                for other_tsg in rxn.ts_species.ts_guesses:
                    if almost_equal_coords(xyz, other_tsg.initial_xyz):
                        if 'heuristics' not in other_tsg.method.lower():
                            other_tsg.method += ' and Heuristics'
                        unique = False
                        break
                if unique:
                    ts_guess = TSGuess(method=f'Heuristics',
                                       index=len(rxn.ts_species.ts_guesses),
                                       method_index=method_index,
                                       t0=tsg.t0,
                                       execution_time=tsg.execution_time,
                                       success=True,
                                       family=family,
                                       xyz=xyz,
                                       )
                    rxn.ts_species.ts_guesses.append(ts_guess)
                    save_geo(xyz=xyz,
                             path=self.local_path,
                             filename=f'Heuristics {method_index}',
                             format_='xyz',
                             comment=f'Heuristics {method_index}, family: {family}',
                             )

            if len(self.reactions) < 5:
                successes = len([tsg for tsg in rxn.ts_species.ts_guesses if tsg.success and 'heuristics' in tsg.method])
                if successes:
                    logger.info(f'Heuristics successfully found {successes} TS guesses for {rxn.label}.')
                else:
                    logger.info(f'Heuristics did not find any successful TS guesses for {rxn.label}.')

        self.final_time = datetime.datetime.now()

    def execute_queue(self):
        """
        (Execute a job to the server's queue.)
        A single Heuristics job will always be executed incore.
        """
        self.execute_incore()


def combine_coordinates_with_redundant_atoms(xyz1,
                                             xyz2,
                                             mol1,
                                             mol2,
                                             h1,
                                             h2,
                                             c=None,
                                             d=None,
                                             r1_stretch=1.2,
                                             r2_stretch=1.2,
                                             a2=180,
                                             d2=None,
                                             d3=None,
                                             keep_dummy=False,
                                             ) -> dict:
    """
    Combine two coordinates that share an atom.
    For this redundant atom case, only three additional degrees of freedom (here ``a2``, ``d2``, and ``d3``)
    are required. The number of atoms in ``mol2`` should be lesser than or equal to the number of atoms in ``mol1``.

    Atom scheme (dummy atom X will be added if the A-H-B angle is close to 180 degrees)::

                    X           D
                    |         /
            A -- H1 - H2 -- B
          /
        C
        |--- mol1 --|-- mol2 ---|

    zmats will be constructed in the following way::

        zmat1 = {'symbols': ('C', 'H', 'H', 'H', 'H'),
                 'coords': ((None, None, None),  # 0, atom A
                            ('R_1_0', None, None),  # 1, atom C
                            ('R_2_1', 'A_2_1_0', None),  # 2
                            ('R_3_2', 'A_3_2_0', 'D_3_2_0_1'),  # 3
                            ('R_4_3' * r1_stretch, 'A_4_3_2', 'D_4_3_2_1')),  # 4, H1
                 'vars': {...},
                 'map': {...}}

        zmat2 = {'symbols': ('H', 'H', 'H', 'H', 'C'),
                 'coords': ((None, None, None),  # H2, redundant H atom, will be united with H1
                            ('R_5_4' * r2_stretch, a2 (B-H-A) = 'A_5_4_0', d2 (B-H-A-C) = 'D_5_4_0_1'),  # 5, atom B
                            ('R_6_5', 'A_6_5_4', d3 (D-B-H-C) = 'D_6_5_4_1'),  # 6, atom D
                            ('R_7_6', 'A_7_6_4', 'D_7_6_4_5'),  # 7
                            ('R_8_7', 'A_8_7_6', 'D_8_7_6_5')),  # 8
                 'vars': {...},
                 'map': {...}}

    Args:
        xyz1 (dict, str): The Cartesian coordinates of molecule 1 (including the redundant atom).
        xyz2 (dict, str): The Cartesian coordinates of molecule 2 (including the redundant atom).
        mol1 (Molecule): The Molecule instance corresponding to ``xyz1``.
        mol2 (Molecule): The Molecule instance corresponding to ``xyz2``.
        h1 (int): The 0-index of a terminal redundant H atom in ``xyz1`` (atom H1).
        h2 (int): The 0-index of a terminal redundant H atom in ``xyz2`` (atom H2).
        c (int, optional): The 0-index of an atom in ``xyz1`` connected to either A or H1 which is neither A nor H1
                           (atom C).
        d (int, optional): The 0-index of an atom in ``xyz2`` connected to either B or H2 which is neither B nor H2
                           (atom D).
        r1_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h1`` in ``xyz1`` (bond A-H1).
        r2_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h2`` in ``xyz2`` (bond B-H2).
        a2 (float, optional): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        d2 (float, optional): The dihedral angle (in degrees) between atoms B-H-A-C (dihedral B-H-A-C).
                              This argument must be given only if the a2 angle is not linear,
                              and mol2 has 3 or more atoms, otherwise it is meaningless.
        d3 (float, optional): The dihedral angel (in degrees) between atoms D-B-H-C (dihedral D-B-H-C).
                              This parameter is mandatory only if atom D exists (i.e., if ``mol2`` has 3 or more atoms).
        keep_dummy (bool, optional): Whether to keep a dummy atom if added, ``True`` to keep, ``False`` by default.

    Returns:
        dict: The combined Cartesian coordinates.

    Todo:
        - Accept xyzs of the radicals as well as E0's of all species, average xyz of atoms by energy similarity
          before returning the final cartesian coordinates
    """
    is_a2_linear = is_angle_linear(a2)

    if len(mol1.atoms) == 1 or len(mol2.atoms) == 1:
        raise ValueError(
            f'The molecule arguments to combine_coordinates_with_redundant_atoms must each have more than 1 '
            f'atom (including the abstracted hydrogen atom in each), got {len(mol1.atoms)} atoms in mol1 '
            f'and {len(mol2.atoms)} atoms in mol2.')
    if not is_a2_linear and len(mol1.atoms) > 2 and d2 is None:
        raise ValueError('The d2 parameter (the B-H-A-C dihedral) must be given if the a2 angle (B-H-A) is not close '
                         'to 180 degrees, got None.')
    if is_angle_linear(a2) and d2 is not None:
        logger.warning(f'The combination a2={a2} and d2={d2} is meaningless (cannot rotate a dihedral about a linear '
                       f'angle). Not considering d2.')
        d2 = None
    if len(mol1.atoms) > 2 and c is None:
        raise ValueError('The c parameter (the index of atom C in xyz1) must be given if mol1 has 3 or more atoms, '
                         'got None.')
    if len(mol2.atoms) > 2 and d is None:
        raise ValueError('The d parameter (the index of atom D in xyz2) must be given if mol2 has 3 or more atoms, '
                         'got None.')
    if len(mol2.atoms) > 2 and d3 is None:
        raise ValueError('The d3 parameter (dihedral D-B-H-C) must be given if mol2 has 3 or more atoms, got None.')

    a = mol1.atoms.index(list(mol1.atoms[h1].edges.keys())[0])
    b = mol2.atoms.index(list(mol2.atoms[h2].edges.keys())[0])
    if c is not None and c == a:
        raise ValueError(f'The value for c ({c}) is invalid (it represents atom A, not atom C)')
    if c is not None and d == b:
        raise ValueError(f'The value for d ({d}) is invalid (it represents atom B, not atom D)')

    # generate the two constrained zmats
    constraints1 = {'R_atom': [(h1, a)]}
    zmat1 = zmat_from_xyz(xyz=xyz1, mol=mol1, constraints=constraints1, consolidate=False)

    constraints2 = {'A_group': [(d, b, h2)]} if d is not None else {'R_group': [(b, h2)]}
    zmat2 = zmat_from_xyz(xyz=xyz2, mol=mol2, constraints=constraints2, consolidate=False)

    # stretch the A--H1 and B--H2 bonds
    r_a_h1_param = get_parameter_from_atom_indices(zmat=zmat1, indices=(h1, a), xyz_indexed=True)
    r_b_h2_param = get_parameter_from_atom_indices(zmat=zmat2, indices=(b, h2), xyz_indexed=True)
    zmat1['vars'][r_a_h1_param] *= r1_stretch
    zmat2['vars'][r_b_h2_param] *= r2_stretch

    # determine the "glue" parameters
    num_atoms_1 = len(zmat1['symbols'])  # the number of atoms in zmat1, used to increment the atom indices in zmat2
    zh = num_atoms_1 - 1  # the atom index of H in the combined zmat
    za = key_by_val(zmat1['map'], a)  # the atom index of A in the combined zmat
    zb = num_atoms_1 + int(is_a2_linear)  # the atom index of B in the combined zmat, if a2=180, consider the dummy atom
    zc = key_by_val(zmat1['map'], c) if c is not None else None
    zd = num_atoms_1 + 1 + int(is_a2_linear) if d is not None else None  # the atom index of B in the combined zmat
    param_a2 = f'A_{zb}_{zh}_{za}'  # B-H-A
    param_d2 = f'D_{zb}_{zh}_{za}_{zc}' if zc is not None else None  # B-H-A-C
    if is_a2_linear:
        # add a dummy atom
        zx = num_atoms_1
        num_atoms_1 += 1
        zmat1['symbols'] = tuple(list(zmat1['symbols']) + ['X'])
        r_str = f'RX_{zx}_{zh}'
        a_str = f'AX_{zx}_{zh}_{za}'
        d_str = f'DX_{zx}_{zh}_{za}_{zc}' if zc is not None else None  # X-H-A-C
        zmat1['coords'] = tuple(list(zmat1['coords']) + [(r_str, a_str, d_str)])  # the coords of the dummy atom
        zmat1['vars'][r_str] = 1.0
        zmat1['vars'][a_str] = 90.0
        if d_str is not None:
            zmat1['vars'][d_str] = 0
        param_a2 = f'A_{zb}_{zh}_{zx}'  # B-H-X
        param_d2 = f'D_{zb}_{zh}_{zx}_{za}' if zc is not None else None  # B-H-X-A
    if d3 is not None and zd is not None:
        param_d3 = f'D_{zd}_{zb}_{zh}_{zc}'  # D-B-H-C
    else:
        param_d3 = None

    # generate a modified zmat2: remove the first atom, change all existing parameter indices, add "glue" parameters
    new_coords, new_vars = list(), dict()
    for i, coords in enumerate(zmat2['coords'][1:]):
        new_coord = list()
        for j, param in enumerate(coords):
            if param is not None:
                if i == 0 and is_a2_linear:
                    # atom B should refer to H, not X
                    new_param = up_param(param=param, increment_list=[num_atoms_1 - 1, num_atoms_1 - 2])
                else:
                    new_param = up_param(param=param, increment=num_atoms_1 - 1)
                new_coord.append(new_param)
                new_vars[new_param] = zmat2['vars'][param]  # keep the original parameter R/A/D values
            else:
                if i == 0 and j == 1:
                    # this is a2
                    new_coord.append(param_a2)
                    new_vars[param_a2] = a2 + 90 if is_a2_linear else a2
                elif i == 0 and j == 2 and c is not None:
                    # this is d2
                    new_coord.append(param_d2)
                    new_vars[param_d2] = 0 if is_a2_linear else d2
                elif i == 1 and j == 2 and param_d3 is not None:
                    # this is d3
                    new_coord.append(param_d3)
                    new_vars[param_d3] = d3
                else:
                    new_coord.append(None)
        new_coords.append(tuple(new_coord))

    combined_zmat = dict()
    combined_zmat['symbols'] = tuple(zmat1['symbols'] + zmat2['symbols'][1:])
    combined_zmat['coords'] = tuple(list(zmat1['coords']) + new_coords)
    combined_zmat['vars'] = {**zmat1['vars'], **new_vars}  # combine the two dicts
    combined_zmat['map'] = dict()
    x_occurrences = 0
    for i, symbol in enumerate(combined_zmat['symbols']):
        if symbol == 'X':
            combined_zmat['map'][i] = 'X'
            x_occurrences += 1
        else:
            combined_zmat['map'][i] = i - x_occurrences

    for i, coords in enumerate(combined_zmat['coords']):
        if i > 2 and None in coords:
            raise ValueError(f'Could not combine zmats, got a None parameter above the 3rd row:\n{combined_zmat}')
    return zmat_to_xyz(zmat=combined_zmat, keep_dummy=keep_dummy)


def label_molecules(reactants: List['Molecule'],
                    products: List['Molecule'],
                    family: 'KineticsFamily',
                    output_with_resonance: bool = False,
                    ) -> Optional[Reaction]:
    """
    React molecules to give the requested products via an RMG family.
    Results in a reaction with RMG's atom labels for the reactants and products.

    Args:
        reactants (List['Molecule']): Entries are Molecule instances of the reaction reactants.
        products (List['Molecule']): Entries are Molecule instances of the reaction products.
        family ('KineticsFamily'): The RMG reaction family instance.
        output_with_resonance (bool, optional): Whether to generate all resonance structures with labels.
                                                ``True`` to generate, ``False`` by default.

    Returns:
        Optional[Reaction]: An RMG Reaction instance with atom-labeled reactants and products.
    """
    reaction = Reaction(reactants=reactants, products=products)
    try:
        family.add_atom_labels_for_reaction(reaction=reaction,
                                            output_with_resonance=output_with_resonance,
                                            save_order=True,
                                            )
    except (ActionError, ValueError):
        return None
    return reaction


# Family-specific heuristics:


def h_abstraction(arc_reaction: 'ARCReaction',
                  rmg_reactions: List['Reaction'],
                  r1_stretch: float = 1.2,
                  r2_stretch: float = 1.2,
                  a2: float = 180,
                  dihedral_increment: int = 20,
                  ) -> List[dict]:
    """
    Generate TS guesses for reactions of the RMG H_Abstraction family.

    Args:
        arc_reaction: An ARCReaction instance.
        rmg_reactions: Entries are RMGReaction instances. The reactants and products attributes should not contain
                       resonance structures as only the first molecule is consider - pass several Reaction entries
                       instead. Atoms must be labeled according to the RMG reaction family.
        r1_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h1`` in ``xyz1`` (bond A-H1).
        r2_stretch (float, optional): The factor by which to multiply (stretch/shrink) the bond length to the terminal
                                      atom ``h2`` in ``xyz2`` (bond B-H2).
        a2 (float, optional): The angle (in degrees) in the combined structure between atoms B-H-A (angle B-H-A).
        dihedral_increment (int, optional): The dihedral increment to use for B-H-A-C and D-B-H-C dihedral scans.

    Returns: List[dict]
        Entries are Cartesian coordinates of TS guesses for all reactions.
    """
    xyz_guesses = list()

    # Identify R1H and R2H in the "R1H + R2 <=> R1 + R2H" or "R2 + R1H <=> R2H + R1" reaction
    # using the first RMG reaction; all other RMG reactions and the ARC reaction should have the same order.
    reactants_reversed, products_reversed = False, False
    for i, reactant in enumerate(rmg_reactions[0].reactants):
        for atom in reactant.molecule[0].atoms:
            if atom.label == '*2' and i != 0:
                reactants_reversed = True
    for i, product in enumerate(rmg_reactions[0].products):
        for atom in product.molecule[0].atoms:
            if atom.label == '*2' and i != 1:
                products_reversed = True

    arc_reactant = arc_reaction.r_species[1] if reactants_reversed else arc_reaction.r_species[0]
    arc_product = arc_reaction.p_species[0] if products_reversed else arc_reaction.p_species[1]

    for rmg_reaction in rmg_reactions:
        rmg_reactant_mol = rmg_reaction.reactants[1].molecule[0] if reactants_reversed \
            else rmg_reaction.reactants[0].molecule[0]
        rmg_product_mol = rmg_reaction.products[0].molecule[0] if products_reversed \
            else rmg_reaction.products[1].molecule[0]

        h1 = rmg_reactant_mol.atoms.index([atom for atom in rmg_reactant_mol.atoms
                                           if atom.label == '*2'][0])
        h2 = rmg_product_mol.atoms.index([atom for atom in rmg_product_mol.atoms
                                          if atom.label == '*2'][0])

        c = find_distant_neighbor(rmg_mol=rmg_reactant_mol, start=h1)
        d = find_distant_neighbor(rmg_mol=rmg_product_mol, start=h2)
        mol = rmg_product_mol.copy()

        # d2 describes the B-H-A-C dihedral, populate d2_values if C exists and the B-H-A angle is not linear.
        d2_values = list(range(0, 360, dihedral_increment)) if len(rmg_reactant_mol.atoms) > 2 \
            and not is_angle_linear(a2) else list()

        # d3 describes the D-B-H-C dihedral, populate d3_values if D and C exist.
        d3_values = list(range(0, 360, dihedral_increment)) if len(rmg_product_mol.atoms) > 2 \
            and len(rmg_product_mol.atoms) > 2 else list()

        if d2_values and d3_values:
            d2_d3_product = list(itertools.product(d2_values, d3_values))
        elif d2_values:
            d2_d3_product = [(d2, None) for d2 in d2_values]
        elif d3_values:
            d2_d3_product = [(None, d3) for d3 in d3_values]
        else:
            d2_d3_product = [(None, None)]
        # Todo:
        # r1_stretch_, r2_stretch_, a2_ = get_training_params(
        #     family='H_Abstraction',
        #     atom_type_key=tuple(sorted([atom_a.atomtype.label, atom_b.atomtype.label])),
        #     atom_symbol_key=tuple(sorted([atom_a.element.symbol, atom_b.element.symbol])),
        # )
        # r1_stretch_, r2_stretch_, a2_ = 1.2, 1.2, 170  # general guesses

        zmats = list()
        for d2, d3 in d2_d3_product:
            xyz_guess = None
            try:
                xyz_guess = combine_coordinates_with_redundant_atoms(xyz1=arc_reactant.get_xyz(),
                                                                     xyz2=arc_product.get_xyz(),
                                                                     mol1=rmg_reactant_mol,
                                                                     mol2=rmg_product_mol,
                                                                     h1=h1,
                                                                     h2=h2,
                                                                     c=c,
                                                                     d=d,
                                                                     r1_stretch=r1_stretch,
                                                                     r2_stretch=r2_stretch,
                                                                     a2=a2,
                                                                     d2=d2,
                                                                     d3=d3,
                                                                     )
            except ValueError as e:
                logger.error(f'Could not generate a guess using Heuristics for H abstraction reaction, got:\n{e}')

            if xyz_guess is not None and not colliding_atoms(xyz_guess):
                # len(qcel.molutil.guess_connectivity(symbols, geometry, threshold=0.9))
                zmat_guess = zmat_from_xyz(xyz_guess)
                for existing_zmat_guess in zmats:
                    if compare_zmats(existing_zmat_guess, zmat_guess):
                        break
                else:
                    # This TS is unique, and has no atom collisions.
                    zmats.append(zmat_guess)
                    xyz_guess = reverse_xyz(xyz_guess, reactants_reversed, rmg_reactant_mol)
                    xyz_guesses.append(xyz_guess)

    # Todo: Learn bond stretches and the A-H-B angle for different atom types.
    return xyz_guesses


def reverse_xyz(xyz: dict,
                reactants_reversed: bool,
                rmg_reactant_mol: 'Molecule',
                ) -> dict:
    """
    Sort the atoms order in a TS xyz guess according to the reactants order in the reaction.

    Args:
        xyz (dict): The TS xyz guess.
        reactants_reversed (bool): Whether the reactants were reversed when generating the TS guess.
        rmg_reactant_mol ('Molecule'): The Molecule object instance describing the first reactant in the
                                       according to the RMG family template.

    Returns:
        dict: The sorted TS xyz guess.
    """
    if not reactants_reversed:
        return xyz
    r1_atoms_num = len(rmg_reactant_mol.atoms)
    r2_atoms_num = len(xyz['symbols']) - r1_atoms_num
    symbols, coords, isotopes = list(), list(), list()
    for i in range(r1_atoms_num):
        symbols.append(xyz['symbols'][i + r2_atoms_num])
        coords.append(xyz['coords'][i + r2_atoms_num])
        isotopes.append(xyz['isotopes'][i + r2_atoms_num])
    for i in range(r2_atoms_num):
        symbols.append(xyz['symbols'][i])
        coords.append(xyz['coords'][i])
        isotopes.append(xyz['isotopes'][i])
    return xyz_from_data(symbols=symbols, coords=coords, isotopes=isotopes)


def find_distant_neighbor(rmg_mol: 'Molecule',
                          start: int,
                          ) -> Optional[int]:
    """
    Find the 0-index of a distant neighbor (2 steps away) if possible from the start atom.
    Preferably, a heavy atom will be returned.

    Args:
        rmg_mol ('Molecule'): The RMG molecule object instance to explore.
        start (int): The 0-index of the start atom.

    Returns:
        Optional[int]: The 0-index of the distant neighbor.
    """
    if len(rmg_mol.atoms) <= 2:
        return None
    distant_neighbor_h_index = None
    for neighbor in rmg_mol.atoms[start].edges.keys():
        for distant_neighbor in neighbor.edges.keys():
            distant_neighbor_index = rmg_mol.atoms.index(distant_neighbor)
            if distant_neighbor_index != start:
                if distant_neighbor.is_hydrogen():
                    distant_neighbor_h_index = distant_neighbor_index
                else:
                    return distant_neighbor_index
    return distant_neighbor_h_index


register_job_adapter('heuristics', HeuristicsAdapter)
