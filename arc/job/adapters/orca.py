"""
An adapter for executing Orca jobs

https://orcaforum.kofo.mpg.de/app.php/portal
"""

import datetime
import math
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from mako.template import Template

from arc.common import get_logger
from arc.imports import incore_commands, settings
from arc.job.adapter import JobAdapter
from arc.job.adapters.common import (check_argument_consistency,
                                     is_restricted,
                                     set_job_args,
                                     update_input_dict_with_args,
                                     which)
from arc.job.factory import register_job_adapter
from arc.job.local import execute_command, submit_job
from arc.job.ssh import SSHClient
from arc.settings.settings import orca_default_options_dict
from arc.species.converter import xyz_to_str

if TYPE_CHECKING:
    from arc.level import Level
    from arc.reaction import ARCReaction
    from arc.species import ARCSpecies

logger = get_logger()

default_job_settings, global_ess_settings, input_filenames, output_filenames, rotor_scan_resolution, servers, \
    submit_filenames = settings['default_job_settings'], settings['global_ess_settings'], settings['input_filenames'], \
                       settings['output_filenames'], settings['rotor_scan_resolution'], settings['servers'], \
                       settings['submit_filenames']

# job_type_1: 'SP', 'Opt', 'OptTS', 'Freq'
# job_type_2: reserved for Opt + Freq.
# restricted: 'R' = closed-shell SCF, 'U' = spin unrestricted SCF, 'RO' = open-shell spin restricted SCF
# auxiliary_basis: required for DLPNO calculations (speed up calculation)
# memory: MB per core (must increase as system gets larger)
# cpus: must be less than number of electron pairs, defaults to min(heavy atoms, cpus limit)
# job_options_blocks: input blocks that enable detailed control over program
# job_options_keywords: input keywords that control the job
# method_class: 'HF' for wavefunction methods (hf, mp, cc, dlpno ...). 'KS' for DFT methods.
# options: additional keywords to control job (e.g., TightSCF, NormalPNO ...)
input_template = """!${restricted}${method_class} ${method} ${basis} ${auxiliary_basis} ${keywords}
! NRSCF # using Newton–Raphson SCF algorithm 
!${job_type_1} 
${job_type_2}
%maxcore ${memory}
%pal # job parallelization settings
nprocs ${cpus}
end
%scf # recommended SCF settings 
NRMaxIt 400
NRStart 0.00005
MaxIter 500
end
${block}

* xyz ${charge} ${multiplicity}
${xyz}
*
"""


class OrcaAdapter(JobAdapter):
    """
    A class for executing Orca jobs.

    Args:
        project (str): The project's name. Used for setting the remote path.
        project_directory (str): The path to the local project directory.
        job_type (list, str): The job's type, validated against ``JobTypeEnum``. If it's a list, pipe.py will be called.
        level (Level): The level of theory to use.
        args (dict, optional): Methods (including troubleshooting) to be used in input files.
                               Keys are either 'keyword', 'block', or 'trsh', values are dictionaries with values
                               to be used either as keywords or as blocks in the respective software input file.
                               If 'trsh' is specified, an action might be taken instead of appending a keyword or a
                               block to the input file (e.g., change server or change scan resolution).
        bath_gas (str, optional): A bath gas. Currently only used in OneDMin to calculate L-J parameters. Allowed values
                                  are: ``'He'``, ``'Ne'``, ``'Ar'``, ``'Kr'``, ``'H2'``, ``'N2'``, or ``'O2'``.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        constraints (list, optional): A list of constraints to use during an optimization or scan.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        dihedrals (List[float], optional): The dihedral angels corresponding to self.torsions.
        ess_settings (dict, optional): A dictionary of available ESS and a corresponding server list.
        ess_trsh_methods (List[str], optional): A list of troubleshooting methods already tried out.
        execution_type (str, optional): The execution type, 'incore', 'queue', or 'pipe'.
        fine (bool, optional): Whether to use fine geometry optimization parameters. Default: ``False``.
        initial_time (datetime.datetime, optional): The time at which this job was initiated.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).
        job_id (int, optional): The job's ID determined by the server.
        job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        job_name (str, optional): The job's name (e.g., 'opt_a103').
        job_num (int, optional): Used as the entry number in the database, as well as in ``job_name``.
        job_status (int, optional): The job's server and ESS statuses.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        reactions (List[ARCReaction], optional): Entries are ARCReaction instances, used for TS search methods.
        rotor_index (int, optional): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        server (str): The server to run on.
        server_nodes (list, optional): The nodes this job was previously submitted to.
        species (List[ARCSpecies], optional): Entries are ARCSpecies instances.
                                              Either ``reactions`` or ``species`` must be given.
        tasks (int, optional): The number of tasks to use in a job array (each task has several threads).
        testing (bool, optional): Whether the object is generated for testing purposes, ``True`` if it is.
        torsions (List[List[int]], optional): The 0-indexed atom indices of the torsions identifying this scan point.
    """

    def __init__(self,
                 project: str,
                 project_directory: str,
                 job_type: Optional[Union[List[str], str]] = None,
                 level: Optional['Level'] = None,
                 args: Optional[dict] = None,
                 bath_gas: Optional[str] = None,
                 checkfile: Optional[str] = None,
                 constraints: Optional[List[Tuple[List[int], float]]] = None,
                 cpu_cores: Optional[str] = None,
                 dihedrals: Optional[List[float]] = None,
                 ess_settings: Optional[dict] = None,
                 ess_trsh_methods: Optional[List[str]] = None,
                 execution_type: Optional[str] = None,
                 fine: bool = False,
                 initial_time: Optional['datetime.datetime'] = None,
                 irc_direction: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_memory_gb: float = 14.0,
                 job_name: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_status: Optional[List[Union[dict, str]]] = None,
                 max_job_time: Optional[float] = None,
                 reactions: Optional[List['ARCReaction']] = None,
                 rotor_index: Optional[int] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 species: Optional[List['ARCSpecies']] = None,
                 tasks: Optional[int] = None,
                 testing: bool = False,
                 torsions: List[List[int]] = None,
                 ):

        self.job_adapter = 'orca'
        self.execution_type = execution_type or 'queue'
        self.command = 'orca'
        self.url = 'https://orcaforum.kofo.mpg.de/app.php/portal'

        if species is None:
            raise ValueError('Cannot execute Orca without an ARCSpecies object.')

        if any(arg is None for arg in [job_type, level]):
            raise ValueError(f'All of the following arguments must be given:\n'
                             f'job_type, level, project, project_directory\n'
                             f'Got: {job_type} {level}, respectively')

        self.project = project
        self.project_directory = project_directory
        if self.project_directory and not os.path.isdir(self.project_directory):
            os.makedirs(self.project_directory)
        self.job_types = job_type if isinstance(job_type, list) else [job_type]  # always a list
        self.job_type = job_type if isinstance(job_type, str) else job_type[0]  # always a string
        self.level = level
        self.args = args or dict()
        self.bath_gas = bath_gas
        self.checkfile = checkfile
        self.constraints = constraints or list()
        self.cpu_cores = cpu_cores
        self.dihedrals = dihedrals
        self.ess_settings = ess_settings or global_ess_settings
        self.ess_trsh_methods = ess_trsh_methods or list()
        self.fine = fine
        self.initial_time = datetime.datetime.strptime(initial_time, '%Y-%m-%d %H:%M:%S') \
            if isinstance(initial_time, str) else initial_time
        self.irc_direction = irc_direction
        self.job_id = job_id
        self.job_memory_gb = job_memory_gb
        self.job_name = job_name
        self.job_num = job_num
        self.job_status = job_status \
            or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.reactions = [reactions] if not isinstance(reactions, list) else reactions
        self.rotor_index = rotor_index
        self.server = server
        self.server_nodes = server_nodes or list()
        self.species = [species] if not isinstance(species, list) else species
        self.tasks = tasks
        self.testing = testing
        self.torsions = torsions

        if self.job_num is None:
            self._set_job_number()
            self.job_name = f'{self.job_type}_a{self.job_num}'

        self.args = set_job_args(args=self.args, level=self.level, job_name=self.job_name)

        self.final_time = None
        self.charge = self.species[0].charge
        self.multiplicity = self.species[0].multiplicity
        self.is_ts = self.species[0].is_ts
        self.run_time = None
        self.scan_res = self.args['trsh']['scan_res'] if 'scan_res' in self.args['trsh'] else rotor_scan_resolution

        self.server = self.args['trsh']['server'] if 'server' in self.args['trsh'] \
            else self.ess_settings[self.job_adapter][0] if isinstance(self.ess_settings[self.job_adapter], list) \
            else self.ess_settings[self.job_adapter]
        self.species_label = self.species[0].label

        self.cpu_cores, self.input_file_memory, self.submit_script_memory = None, None, None
        self.set_cpu_and_mem()
        self.set_file_paths()

        self.tasks = None
        self.iterate_by = list()
        self.number_of_processes = 0
        self.determine_job_array_parameters()  # Writes the local HDF5 file if needed.

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
        input_dict = dict()
        for key in ['block',
                    'job_type_1',
                    'job_type_2',
                    'keywords',
                    ]:
            input_dict[key] = ''
        input_dict['auxiliary_basis'] = self.level.auxiliary_basis or ''
        input_dict['basis'] = self.level.basis or ''
        input_dict['charge'] = self.charge
        input_dict['cpus'] = self.cpu_cores
        input_dict['label'] = self.species_label
        input_dict['memory'] = self.input_file_memory
        input_dict['method'] = self.level.method
        input_dict['multiplicity'] = self.multiplicity
        input_dict['xyz'] = xyz_to_str(self.species[0].get_xyz())

        scf_convergence = self.args['keyword'].get('scf_convergence', '').lower() or \
            orca_default_options_dict['global']['keyword'].get('scf_convergence', '').lower()
        if not scf_convergence:
            raise ValueError('Orca SCF convergence is not specified. Please specify this variable either in '
                             'settings.py as default or in the input file as additional options.')
        self.add_to_args(val=scf_convergence, key1='keyword')

        # Orca requires different blocks for wavefunction methods and DFT methods
        if self.level.method_type == 'dft':
            input_dict['method_class'] = 'KS'
            # DFT grid must be the same for both opt and freq
            if self.fine:
                self.add_to_args(val='Grid6 NoFinalGrid', key1='keyword')
            else:
                self.add_to_args(val='Grid5 NoFinalGrid', key1='keyword')
        elif self.level.method_type == 'wavefunction':
            input_dict['method_class'] = 'HF'
            if 'dlpno' in self.level.method:
                dlpno_threshold = self.args['keyword'].get('dlpno_threshold', '').lower() or \
                    orca_default_options_dict['global']['keyword'].get('dlpno_threshold', '').lower()
                if not dlpno_threshold:
                    raise ValueError('Orca DLPNO threshold is not specified. Please specify this variable either in '
                                     'settings.py as default or in the input file as additional options.')
                self.add_to_args(val=dlpno_threshold, key1='keyword')
        else:
            logger.debug(f'Running {self.level.method_type} method {self.level.method} method in Orca.')

        input_dict['restricted'] = 'r' if is_restricted(self) else 'u'

        # Job type specific options
        if self.job_type in ['opt', 'conformers', 'optfreq']:
            opt_convergence_key = 'fine_opt_convergence' if self.fine else 'opt_convergence'
            opt_convergence = self.args['keyword'].get(opt_convergence_key, '').lower() or \
                orca_default_options_dict['opt']['keyword'].get(opt_convergence_key, '').lower()
            if not opt_convergence:
                raise ValueError('Orca optimization convergence (NormalOpt or TightOpt) is not specified. '
                                 'Please specify this variable either in the settings.py as default options '
                                 'or in the input file as additional options.')
            self.add_to_args(val=opt_convergence, key1='keyword')
            if not self.is_ts:
                input_dict['job_type_1'] = 'Opt'
            else:
                input_dict['job_type_1'] = 'OptTS'
                self.add_to_args(val="""
%geom
Calc_Hess true # calculation of the exact Hessian before the first opt step
end               
""",
                                 key1='block')

        elif self.job_type in ['freq', 'optfreq']:
            if self.job_type == 'freq':
                input_dict['job_type_1'] = 'Freq'
            elif self.job_type == 'optfreq':
                input_dict['job_type_2'] = '!Freq'
            use_num_freq = self.args['keyword'].get('use_num_freq', False) \
                or orca_default_options_dict['freq']['keyword'].get('use_num_freq', False)
            if use_num_freq:
                self.add_to_args(val='NumFreq', key1='keyword')
                logger.info(f'Using numerical frequencies calculation in Orca. Note: This job might therefore be '
                            f'time-consuming.')

        elif self.job_type == 'sp':
            input_dict['job_type_1'] = 'sp'

        input_dict = update_input_dict_with_args(args=self.args, input_dict=input_dict)

        with open(os.path.join(self.local_path, input_filenames[self.job_adapter]), 'w') as f:
            f.write(Template(input_template).render(**input_dict))

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
        self.files_to_upload, self.files_to_download = list(), list()
        # 1. ** Upload **
        # 1.1. submit file
        if self.execution_type != 'incore':
            # we need a submit file for single or array jobs (either submitted to local or via SSH)
            self.write_submit_script()
            self.files_to_upload.append(self.get_file_property_dictionary(
                file_name=submit_filenames[servers[self.server]['cluster_soft']]))
        # 1.2. input file
        if not self.iterate_by:
            # if this is not a job array, we need the ESS input file
            self.write_input_file()
            self.files_to_upload.append(self.get_file_property_dictionary(file_name=input_filenames[self.job_adapter]))
        # 1.3. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_upload.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        # 2. ** Download **
        # 2.1. HDF5 file
        if self.iterate_by and os.path.isfile(os.path.join(self.local_path, 'data.hdf5')):
            self.files_to_download.append(self.get_file_property_dictionary(file_name='data.hdf5'))
        else:
            # 2.2. log file
            self.files_to_download.append(self.get_file_property_dictionary(
                file_name=output_filenames[self.job_adapter]))
            # 2.3. Hessian file generated by frequency calculations
            # The Hessian file is useful when the user would like to project out the rotors
            if self.job_type in ['freq', 'optfreq']:
                self.files_to_download.append(self.get_file_property_dictionary(file_name='input.hess'))

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
        # Orca's memory is per cpu core and in MB
        self.input_file_memory = math.ceil(self.job_memory_gb * 1024 / self.cpu_cores)

    def execute_incore(self):
        """
        Execute a job incore.
        """
        which(self.command,
              return_bool=True,
              raise_error=True,
              raise_msg=f'Please install {self.job_adapter}, see {self.url} for more information.',
              )
        self._log_job_execution()
        execute_command(incore_commands[self.server][self.job_adapter])

    def execute_queue(self):
        """
        Execute a job to the server's queue.
        """
        self._log_job_execution()
        # submit to queue, differentiate between local (same machine using its queue) and remote servers
        if self.server != 'local':
            with SSHClient(self.server) as ssh:
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
        else:
            # submit to the local queue
            self.job_status[0], self.job_id = submit_job(path=self.local_path)


register_job_adapter('orca', OrcaAdapter)
