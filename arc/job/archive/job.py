"""
The ARC Job module
"""

import csv
import datetime
import math
import os
import shutil
from pprint import pformat
from typing import Dict, Optional, Union

from arc.common import get_logger
from arc.exceptions import JobError, InputError
from arc.job.inputs import input_files
from arc.job.local import (get_last_modified_time,
                           submit_job,
                           delete_job,
                           execute_command,
                           check_job_status,
                           rename_output)
from arc.job.submit import submit_scripts
from arc.job.ssh import SSHClient
from arc.job.trsh import determine_ess_status, trsh_job_on_server
from arc.level import Level
from arc.plotter import save_geo
from arc.settings import (arc_path,
                          default_job_settings,
                          servers,
                          submit_filenames,
                          t_max_format,
                          input_filenames,
                          output_filenames,
                          rotor_scan_resolution,
                          orca_default_options_dict)
from arc.species.converter import check_xyz_dict, xyz_to_str
from arc.species.vectors import calculate_dihedral_angle


logger = get_logger()


class Job(object):
    """
    ARC's Job class.

    Args:
        project (str): The project's name. Used for naming the directory.
        project_directory (str): The path to the project directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_label (str): The species/TS label. Used for naming the directory.
        xyz (dict): The xyz geometry. Used for the calculation.
        job_type (str): The job's type.
        level (Level, dict, str): The level of theory to use.
        multiplicity (int): The species multiplicity.
        charge (int, optional): The species net charge. Default is 0.
        conformer (int, optional): Conformer number if optimizing conformers.
        fine (bool, optional): Whether to use fine geometry optimization parameters.
        shift (str, optional): A string representation alpha- and beta-spin orbitals shifts (molpro only).
        software (str, optional): The electronic structure software to be used.
        is_ts (bool): Whether this species represents a transition structure. Default: ``False``.
        scan (list, optional): A list representing atom labels for the dihedral scan
                              (e.g., "2 1 3 5" as a string or [2, 1, 3, 5] as a list of integers).
        pivots (list, optional): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
                                 but used to identify the rotor.
        total_job_memory_gb (int, optional): The total job allocated memory in GB (14 by default).
        comments (str, optional): Job comments (archived, not used).
        args (str, dict optional): Methods (including troubleshooting) to be used in input files.
                                   Keys are either 'keyword' or 'block', values are dictionaries with values to be used
                                   either as keywords or as blocks in the respective software input file. If ``args``
                                   attribute is given as a string, it will be converted to a dictionary format with
                                   'keyword' and 'general' key.
        scan_trsh (str, optional): A troubleshooting method for rotor scans.
        ess_trsh_methods (list, optional): A list of troubleshooting methods already tried out for ESS convergence.
        bath_gas (str, optional): A bath gas. Currently used in OneDMin to calc L-J parameters.
                                  Allowed values are He, Ne, Ar, Kr, H2, N2, O2
        job_num (int, optional): Used as the entry number in the database, as well as the job name on the server.
        job_server_name (str, optional): Job's name on the server (e.g., 'a103').
        job_name (str, optional): Job's name for internal usage (e.g., 'opt_a103').
        job_id (int, optional): The job's ID determined by the server.
        server (str, optional): Server's name.
        initial_time (datetime, optional): The date-time this job was initiated.
        occ (int, optional): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        max_job_time (float, optional): The maximal allowed job time on the server in hours (can be fractional).
        scan_res (int, optional): The rotor scan resolution in degrees.
        checkfile (str, optional): The path to a previous Gaussian checkfile to be used in the current job.
        number_of_radicals (int, optional): The number of radicals (inputted by the user, ARC won't attempt to
                                            determine it). Defaults to None. Important, e.g., if a Species is a bi-rad
                                            singlet, in which case the job should be unrestricted with
                                            multiplicity = 1.
        conformers (str, optional): A path to the YAML file conformer coordinates for a Gromacs MD job.
        radius (float, optional): The species radius in Angstrom.
        directed_scans (list): Entries are lists of four-atom dihedral scan indices to constrain during a directed scan.
        directed_dihedrals (list): The dihedral angles of a directed scan job corresponding to ``directed_scans``.
        directed_scan_type (str): The type of the directed scan.
        rotor_index (int): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        testing (bool, optional): Whether the object is generated for testing purposes, True if it is.
        cpu_cores (int, optional): The total number of cpu cores requested for a job.
        irc_direction (str, optional): The direction of the IRC job (`forward` or `reverse`).

    Attributes:
        project (str): The project's name. Used for naming the directory.
        ess_settings (dict): A dictionary of available ESS and a corresponding server list.
        species_label (str): The species/TS label. Used for naming the directory.
        charge (int): The species net charge. Default is 0.
        multiplicity (int): The species multiplicity.
        number_of_radicals (int): The number of radicals (inputted by the user, ARC won't attempt to determine it).
                                  Defaults to None. Important, e.g., if a Species is a bi-rad singlet, in which case
                                  the job should be unrestricted with multiplicity = 1.
        spin (int): The spin. automatically derived from the multiplicity.
        xyz (dict): The xyz geometry. Used for the calculation.
        radius (float): The species radius in Angstrom.
        n_atoms (int): The number of atoms in self.xyz.
        conformer (int): Conformer number if optimizing conformers.
        conformers (str): A path to the YAML file conformer coordinates for a Gromacs MD job.
        is_ts (bool): Whether this species represents a transition structure.
        level (Level): The level of theory to use.
        job_type (str): The job's type.
        scan (list): A list representing atom labels for the dihedral scan (e.g., [2, 1, 3, 5]).
        pivots (list): The rotor scan pivots, if the job type is scan. Not used directly in these methods,
                       but used to identify the rotor.
        scan_res (int): The rotor scan resolution in degrees.
        software (str): The electronic structure software to be used.
        server_nodes (list): A list of nodes this job was submitted to (for troubleshooting).
        cpu_cores (int): The total number of cpu cores requested for a job.
                         ARC adopts the following naming system to describe computing hardware hierarchy
                         node > cpu > cpu_cores > cpu_threads
        input_file_memory (int): The memory ARC writes to job input files in appropriate formats per ESS.
                                 In software like Gaussian, this variable is total memory for all cpu cores.
                                 In software like Orca, this variable is memory per cpu core.
        submit_script_memory (int): The memory ARC writes to submit script in appropriate formats per cluster system.
                                    In system like Sun Grid Engine, this variable is total memory for all cpu cores.
                                    In system like Slurm, this variable is memory per cpu core.
                                    Notice that submit_script_memory > input_file_memory because additional memory is
                                    needed to execute a job on server properly
        total_job_memory_gb (int): The total memory ARC specifies for a job in GB.
        fine (bool): Whether to use fine geometry optimization parameters.
        shift (str): A string representation alpha- and beta-spin orbitals shifts (molpro only).
        comments (str): Job comments (archived, not used).
        initial_time (datetime): The date-time this job was initiated.
        final_time (datetime): The date-time this job was initiated.
        run_time (timedelta): Job execution time.
        job_status (list): The job's server and ESS statuses.
                           The job server status is in job.job_status[0] and can be either 'initializing' / 'running'
                           / 'errored' / 'done'. The job ESS status is in job.job_status[1] is a dictionary of
                           {'status': str, 'keywords': list, 'error': str, 'line': str}.
                           The values of 'status' can be either `initializing`, `running`, `errored`, `unconverged`,
                           or `done`. If the status is 'errored', then standardized error keywords, the error
                           description and the identified error line from the ESS log file will be given as well.
        job_server_name (str): Job's name on the server (e.g., 'a103').
        job_name (str): Job's name for internal usage (e.g., 'opt_a103').
        job_id (int): The job's ID determined by the server.
        job_num (int): Used as the entry number in the database, as well as the job name on the server.
        local_path (str): Local path to job's folder.
        local_path_to_output_file (str): The local path to the output.out file.
        local_path_to_orbitals_file (str): The local path to the orbitals.fchk file (only for orbitals jobs).
        local_path_to_check_file (str): The local path to the Gaussian check file of the current job (downloaded).
        local_path_to_lj_file (str): The local path to the lennard_jones data file (from OneDMin).
        local_path_to_xyz (str) The local path to the optimization results file if different than the log file.
        checkfile (str): The path to a previous Gaussian checkfile to be used in the current job.
        remote_path (str): Remote path to job's folder.
        submit (str): The submit script. Created automatically.
        input (str): The input file. Created automatically.
        server (str): Server's name.
        args (dict): Methods (including troubleshooting) to be used in input files. Keys are either 'keyword' or
                     'block', values are dictionaries with values to be used either as keywords or as blocks in the
                     respective software input file.
        ess_trsh_methods (list): A list of troubleshooting methods already tried out for ESS convergence.
        scan_trsh (str): A troubleshooting method for rotor scans.
        occ (int): The number of occupied orbitals (core + val) from a molpro CCSD sp calc.
        project_directory (str): The path to the project directory.
        max_job_time (float): The maximal allowed job time on the server in hours (can be fractional).
        bath_gas (str): A bath gas. Currently used in OneDMin to calc L-J parameters.
                        Allowed values are He, Ne, Ar, Kr, H2, N2, O2.
        directed_scans (list): Entries are lists of four-atom dihedral scan indices to constrain during a directed scan.
        directed_dihedrals (list): The dihedral angles of a directed scan job corresponding to ``directed_scans``.
        directed_scan_type (str): The type of the directed scan.
        rotor_index (int): The 0-indexed rotor number (key) in the species.rotors_dict dictionary.
        irc_direction (str): The direction of the IRC job (`forward` or `reverse`).
    """
    def __init__(self,
                 project: str,
                 project_directory: str,
                 species_label: str,
                 multiplicity: int,
                 job_type: str,
                 level: Union[Level, dict, str],
                 ess_settings: dict,
                 xyz: Optional[dict] = None,
                 charge: int = 0,
                 conformer: int = -1,
                 fine: bool = False,
                 shift: str = '',
                 software: str = None,
                 is_ts: bool = False,
                 scan: Optional[list] = None,
                 pivots: Optional[list] = None,
                 total_job_memory_gb: Optional[int] = None,
                 comments: str = '',
                 args: Optional[Union[Dict[str, Dict[str, str]], str]] = None,
                 scan_trsh: str = '',
                 ess_trsh_methods: Optional[list] = None,
                 bath_gas: Optional[str] = None,
                 job_num: Optional[int] = None,
                 job_server_name: Optional[str] = None,
                 job_name: Optional[str] = None,
                 job_id: Optional[int] = None,
                 job_status: Optional[list] = None,
                 server: Optional[str] = None,
                 server_nodes: Optional[list] = None,
                 initial_time: Optional[Union[datetime.datetime, str]] = None,
                 final_time: Optional[Union[datetime.datetime, str]] = None,
                 occ: Optional[int] = None,
                 max_job_time: Optional[float] = None,
                 scan_res: Optional[int] = None,
                 checkfile: Optional[str] = None,
                 number_of_radicals: Optional[int] = None,
                 conformers: Optional[str] = None,
                 radius: Optional[float] = None,
                 directed_scan_type: Optional[str] = None,
                 directed_scans: Optional[list] = None,
                 directed_dihedrals: Optional[list] = None,
                 rotor_index: Optional[int] = None,
                 testing: bool = False,
                 cpu_cores: Optional[int] = None,
                 irc_direction: Optional[str] = None,
                 ):
        self.project = project
        self.project_directory = project_directory
        self.species_label = species_label
        self.multiplicity = multiplicity
        self.job_type = job_type
        self.level = Level(repr=level)
        self.ess_settings = ess_settings
        self.initial_time = datetime.datetime.strptime(initial_time, '%Y-%m-%d %H:%M:%S') \
            if isinstance(initial_time, str) else initial_time
        self.final_time = datetime.datetime.strptime(final_time, '%Y-%m-%d %H:%M:%S') \
            if isinstance(final_time, str) else final_time
        self.run_time = None
        self.job_num = job_num or -1
        self.charge = charge
        self.number_of_radicals = number_of_radicals
        self.xyz = check_xyz_dict(xyz)
        self.radius = radius
        self.directed_scan_type = directed_scan_type
        self.rotor_index = rotor_index
        self.directed_scans = directed_scans
        self.directed_dihedrals = [float(d) for d in directed_dihedrals] if directed_dihedrals is not None \
            else directed_scans  # it's a string in the restart dict
        self.conformer = conformer
        self.conformers = conformers
        self.is_ts = is_ts
        self.ess_trsh_methods = ess_trsh_methods or list()
        self.args = {'keyword': {'general': args}} if isinstance(args, str) else args or dict()
        for key1 in ['keyword', 'block']:
            if key1 not in self.args:
                self.args[key1] = dict()
        self.scan_trsh = scan_trsh
        self.scan_res = scan_res or rotor_scan_resolution
        self.scan = scan
        self.pivots = pivots or list()
        self.max_job_time = max_job_time or default_job_settings.get('job_time_limit_hrs', 120)
        self.bath_gas = bath_gas
        self.testing = testing
        self.fine = fine
        self.shift = shift
        self.occ = occ
        self.job_status = job_status \
            or ['initializing', {'status': 'initializing', 'keywords': list(), 'error': '', 'line': ''}]
        self.job_id = job_id or 0
        self.comments = comments
        self.checkfile = checkfile
        self.server_nodes = server_nodes or list()
        self.job_server_name = job_server_name
        self.job_name = job_name
        self.server = server
        self.software = software
        if self.software is None:
            self.deduce_software()
        self.cpu_cores = cpu_cores
        self.total_job_memory_gb = total_job_memory_gb or default_job_settings.get('job_total_memory_gb', 14)
        self.irc_direction = irc_direction

        # allowed job types:
        job_types = ['conformer', 'opt', 'freq', 'optfreq', 'sp', 'composite', 'bde', 'scan', 'directed_scan',
                     'gsm', 'irc', 'ts_guess', 'orbitals', 'onedmin', 'ff_param_fit']
        if self.job_type not in job_types:
            raise ValueError(f'Job type {self.job_type} not understood. Must be one of the following:\n{job_types}')
        if self.xyz is None:
            raise InputError(f'{self.job_type} Job of species {self.species_label} got None for xyz')
        if self.job_type == 'directed_scan' and (self.directed_dihedrals is None or self.directed_scans is None
                                                 or self.directed_scan_type is None):
            raise InputError(f'Must have the directed_dihedrals, directed_scans, and directed_scan_type attributes '
                             f'for a directed scan job. Got {self.directed_dihedrals}, {self.directed_scans}, '
                             f'{self.directed_scan_type}, respectively.')
        if self.job_type == 'directed_scan' and self.rotor_index is None:
            raise InputError('Must have the rotor_index argument for a directed scan job.')

        if self.job_num < 0:
            self._set_job_number()
        self.job_server_name = self.job_server_name or 'a' + str(self.job_num)
        if conformer >= 0 and (self.job_name is None or 'conformer_a' in self.job_name):
            if self.job_name is not None:
                logger.warning(f'replacing job name {self.job_name} with conformer_{conformer}')
            self.job_name = f'conformer{conformer}'
        elif self.job_name is None:
            self.job_name = self.job_type + '_' + self.job_server_name

        self.dispersion = ''
        if self.server is None:
            self.server = server or self.ess_settings[self.software][0]

        if self.job_type == 'onedmin' and self.bath_gas is None:
            logger.info(f'Setting bath gas for Lennard-Jones calculation to N2 for species {self.species_label}')
            self.bath_gas = 'N2'
        elif self.bath_gas is not None and self.bath_gas not in ['He', 'Ne', 'Ar', 'Kr', 'H2', 'N2', 'O2']:
            raise InputError(f'Bath gas for OneDMin should be one of the following:\n'
                             f'He, Ne, Ar, Kr, H2, N2, O2.\nGot: {self.bath_gas}')

        self.spin = self.multiplicity - 1
        self.n_atoms = len(self.xyz['symbols']) if self.xyz is not None else None
        self.submit = ''
        self.input = ''
        self.submit_script_memory = None
        self.input_file_memory = None
        self.set_cpu_and_mem()
        self.determine_run_time()

        self.set_file_paths()

        if job_num is None:
            # this checks job_num and not self.job_num on purpose
            # if job_num was given, then don't save as initiated jobs, this is a restarted job
            self._write_initiated_job_to_csv_file()

    def write_submit_script(self):
        """
        Write the Job's submit script.
        """
        un = servers[self.server]['un']  # user name
        size = int(self.radius * 4) if self.radius is not None else None
        if self.max_job_time > 9999 or self.max_job_time <= 0:
            logger.debug('Setting max_job_time to 120 hours')
            self.max_job_time = 120
        t_max = self.format_max_job_time(time_format=t_max_format[servers[self.server]['cluster_soft']])
        architecture = ''
        if self.server.lower() == 'pharos':
            # here we're hard-coding ARC for Pharos, a Green Group server
            # If your server has different node architectures, implement something similar
            if self.cpu_cores <= 8:
                architecture = '\n#$ -l harpertown'
            else:
                architecture = '\n#$ -l magnycours'
        try:
            self.submit = submit_scripts[self.server][self.software.lower()].format(
                name=self.job_server_name, un=un, t_max=t_max, memory=int(self.submit_script_memory),
                cpus=self.cpu_cores, architecture=architecture, size=size)
        except KeyError:
            submit_scripts_for_printing = dict()
            for server, values in submit_scripts.items():
                submit_scripts_for_printing[server] = list()
                for software in values.keys():
                    submit_scripts_for_printing[server].append(software)
            logger.error('Could not find submit script for server {0} and software {1}. Make sure your submit scripts '
                         '(in arc/job/submit.py) are updated with the servers and software defined in arc/settings.py\n'
                         'Alternatively, It is possible that you defined parameters in curly braces (e.g., {{PARAM}}) '
                         'in your submit script/s. To avoid error, replace them with double curly braces (e.g., '
                         '{{{{PARAM}}}} instead of {{PARAM}}.\nIdentified the following submit scripts:\n{2}'.format(
                          self.server, self.software.lower(), submit_scripts_for_printing))
            raise
        if not os.path.isdir(self.local_path):
            os.makedirs(self.local_path)
        with open(os.path.join(self.local_path, submit_filenames[servers[self.server]['cluster_soft']]), 'w') as f:
            f.write(self.submit)
        if self.server != 'local' and not self.testing:
            self._upload_submit_file()

    def write_input_file(self):
        """
        Write a software-specific, job-specific input file.
        Save the file locally and also upload it to the server.
        """
        # Initialize variables
        orca_options_keywords_dict, orca_options_blocks_dict, restricted, method_class = (None for _ in range(4))

        self.input = input_files.get(self.software, None)

        slash, slash_2, scan_string, constraint = '', '', '', ''

        # Determine HF/DFT restriction type
        if (self.multiplicity > 1 and self.level.basis) \
                or (self.number_of_radicals is not None and self.number_of_radicals > 1):
            # run an unrestricted electronic structure calculation if the spin multiplicity is greater than one,
            # or if it is one but the number of radicals is greater than one (e.g., bi-rad singlet)
            # don't run unrestricted for composite methods such as CBS-QB3, it'll be done automatically if the
            # multiplicity is greater than one, but do specify uCBS-QB3 for example for bi-rad singlets.
            if self.number_of_radicals is not None and self.number_of_radicals > 1:
                logger.info(f'Using an unrestricted method for species {self.species_label} which has '
                            f'{self.number_of_radicals} radicals and multiplicity {self.multiplicity}')
            if self.software == 'qchem':
                restricted = 'True'  # In QChem this attribute is "unrestricted"
            elif self.software in ['gaussian', 'orca', 'molpro']:
                restricted = 'u'
        else:
            if self.software == 'qchem':
                restricted = 'False'  # In QChem this attribute is "unrestricted"
            elif self.software == 'orca':
                restricted = 'r'
            else:  # gaussian, molpro
                restricted = ''

        if self.software == 'terachem':
            # TeraChem does not accept "wb97xd3", it expects to get "wb97x" as the method and "d3" as the dispersion
            if self.level.method[-2:] == 'd2':
                self.dispersion = 'd2'
                self.level.method = self.level.method[:-2]
            elif self.level.method[-2:] == 'd3':
                self.dispersion = 'd3'
                self.level.method = self.level.method[:-2]
            elif self.level.method[-1:] == 'd':
                self.dispersion = 'yes'
                self.level.method = self.level.method[:-2]
            else:
                self.dispersion = 'no'
        if self.level.dispersion is not None:
            self.dispersion = self.level.dispersion

        job_type_1, job_type_2, fine = '', '', ''

        # 'vdz' troubleshooting in molpro:
        if self.software == 'molpro' and 'keyword' in self.args and 'trsh' in self.args['keyword'] \
                and 'vdz' in self.args['keyword']['trsh']:
            self.args['keyword']['trsh'] = self.args['keyword']['trsh'].replace('vdz', '')
            self.input = """***,name
memory,{memory},m;
geometry={{angstrom;
{xyz}
}}

basis=cc-pVDZ
int;
{{hf;{shift}
maxit,1000;
wf,spin={spin},charge={charge};}}

{restricted}{method};
{job_type_1}
{job_type_2}
---;"""

        # Software specific global job options
        if self.software == 'orca':
            orca_options_keywords_dict = dict()
            orca_options_blocks_dict = dict()
            user_scf_convergence = ''
            if self.args:
                try:
                    user_block_keywords = self.args['block'].get('general', '')
                except KeyError:
                    user_block_keywords = ''
                if user_block_keywords:
                    orca_options_blocks_dict['global_general'] = user_block_keywords

                try:
                    user_keywords = self.args['keyword'].get('general', '')
                except KeyError:
                    user_keywords = ''
                if user_keywords:
                    orca_options_keywords_dict['global_general'] = user_keywords

                try:
                    user_scf_convergence = self.args['keyword'].get('scf_convergence', '')
                except KeyError:
                    user_scf_convergence = ''
            scf_convergence = user_scf_convergence.lower() or \
                orca_default_options_dict['global']['keyword'].get('scf_convergence', '').lower()
            if not scf_convergence:
                raise InputError('Orca SCF convergence is not specified. Please specify this variable either in the '
                                 'settings.py as default options or in the input file as additional options.')
            orca_options_keywords_dict['scf_convergence'] = scf_convergence

            # Orca requires different job_options_blocks to wavefunction methods and DFTs
            # determine model chemistry type
            if self.level.method_type == 'dft':
                method_class = 'KS'
                # DFT grid must be the same for both opt and freq
                orca_options_keywords_dict['dft_final_grid'] = 'NoFinalGrid'
                if self.fine:
                    orca_options_keywords_dict['dft_grid'] = 'Grid6'
                else:
                    orca_options_keywords_dict['dft_grid'] = 'Grid5'
            elif self.level.method_type == 'wavefunction':
                method_class = 'HF'
                if 'dlpno' in self.level.method:
                    user_dlpno_threshold = ''
                    if self.args:
                        try:
                            user_dlpno_threshold = self.args['keyword'].get('dlpno_threshold', '')
                        except KeyError:
                            user_dlpno_threshold = ''
                    dlpno_threshold = user_dlpno_threshold.lower() if user_dlpno_threshold \
                        else orca_default_options_dict['global']['keyword'].get('dlpno_threshold', '').lower()
                    if not dlpno_threshold:
                        raise InputError(
                            'Orca DLPNO threshold is not specified. Please specify this variable either in the '
                            'settings.py as default options or in the input file as additional options.')
                    orca_options_keywords_dict['dlpno_threshold'] = dlpno_threshold
            else:
                logger.debug(f'Running {self.level.method} method in Orca.')

        # Job type specific options
        if self.job_type in ['conformer', 'opt']:
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, maxcycles=100)'
                else:
                    job_type_1 = 'opt=(calcfc)'
                if self.fine:
                    # Note that the Acc2E argument is not available in Gaussian03
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5, maxcycles=100)'
                    else:
                        job_type_1 = 'opt=(tight, calcfc, maxstep=5)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                if self.fine:
                    fine = '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
                    if self.level.method_type == 'dft':
                        # Try to capture DFT levels, and use a fine DFT grid
                        # See 4.4.5.2 Standard Quadrature Grids, S in
                        # http://www.q-chem.com/qchem-website/manual/qchem50_manual/sect-DFT.html
                        fine += '\n   XC_GRID 3'
            elif self.software == 'molpro':
                if self.is_ts:
                    job_type_1 = "\noptg, root=2, method=qsd, readhess, savexyz='geometry.xyz'"
                else:
                    job_type_1 = "\noptg, savexyz='geometry.xyz'"
            elif self.software == 'orca':
                if self.fine:
                    user_fine_opt_convergence = ''
                    if self.args:
                        try:
                            user_fine_opt_convergence = self.args['keyword'].get('fine_opt_convergence', '')
                        except KeyError:
                            user_fine_opt_convergence = ''
                    fine_opt_convergence = user_fine_opt_convergence.lower() if user_fine_opt_convergence \
                        else orca_default_options_dict['opt']['keyword'].get('fine_opt_convergence', '').lower()
                    if not fine_opt_convergence:
                        raise InputError(
                            'Orca fine optimization convergence is not specified. Please specify this variable either '
                            'in the settings.py as default options or in the input file as additional options.')
                    orca_options_keywords_dict['fine_opt_convergence'] = fine_opt_convergence
                else:
                    user_opt_convergence = ''
                    if self.args:
                        try:
                            user_opt_convergence = self.args['keyword'].get('opt_convergence', '')
                        except KeyError:
                            user_opt_convergence = ''
                    opt_convergence = user_opt_convergence.lower() if user_opt_convergence \
                        else orca_default_options_dict['opt']['keyword'].get('opt_convergence', '').lower()
                    if not opt_convergence:
                        raise InputError(
                            'Orca optimization convergence is not specified. Please specify this variable either in '
                            'the settings.py as default options or in the input file as additional options.')
                    orca_options_keywords_dict['opt_convergence'] = opt_convergence
                if self.is_ts:
                    job_type_1 = 'OptTS'
                    orca_options_blocks_dict['Calc_Hess'] = """
%geom
    Calc_Hess true # calculation of the exact Hessian before the first opt step
end               
"""
                else:
                    job_type_1 = 'Opt'
                self.add_to_args(val=' '.join(orca_options_keywords_dict.values()))
                self.add_to_args(val='\n'.join(orca_options_blocks_dict.values()),
                                 key1='block')
            elif self.software == 'terachem':
                if self.is_ts:
                    raise JobError('TeraChem does not perform TS optimization jobs')
                else:
                    job_type_1 = 'minimize\nnew_minimizer yes'
                if self.fine:
                    fine = '4\ndynamicgrid yes'  # corresponds to ~60,000 grid points/atom
                else:
                    fine = '1'  # default, corresponds to ~800 grid points/atom

        elif self.job_type == 'orbitals' and self.software == 'qchem':
            if self.is_ts:
                job_type_1 = 'ts'
            else:
                job_type_1 = 'opt'
            self.add_to_args(val='\n   NBO           TRUE\n   RUN_NBO6      TRUE\n   '
                                 'PRINT_ORBITALS  TRUE\n   GUI           2',
                             key1='block')

        elif self.job_type == 'ff_param_fit' and self.software == 'gaussian':
            job_type_1, job_type_2 = 'opt', 'freq'
            self.input += """

--Link1--
%chk=check.chk
%mem={memory}mb
%NProcShared={cpus}

# HF/6-31G(d) SCF=Tight Pop=MK IOp(6/33=2,6/41=10,6/42=17) scf(maxcyc=500) guess=read geom=check Maxdisk=2GB

name

{charge} {multiplicity}


"""

        elif self.job_type == 'freq':
            if self.software == 'gaussian':
                job_type_2 = 'freq iop(7/33=1) scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                job_type_1 = 'freq'
            elif self.software == 'molpro':
                job_type_1 = '\n{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'
            elif self.software == 'orca':
                job_type_1 = 'Freq'
                use_num_freq = orca_default_options_dict['freq']['keyword'].get('use_num_freq', False)
                if self.args:
                    try:
                        use_num_freq = self.args['keyword'].get('use_num_freq', False)
                    except KeyError:
                        use_num_freq = False
                if use_num_freq:
                    orca_options_keywords_dict['freq_type'] = 'NumFreq'
                    logger.info(f'Using numerical frequencies calculation in Orca. Note: This job might therefore be '
                                f'time-consuming.')
                self.add_to_args(val=' '.join(orca_options_keywords_dict.values()))
            elif self.software == 'terachem':
                job_type_1 = 'frequencies'

        elif self.job_type == 'optfreq':
            if self.software == 'gaussian':
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, maxstep=5, maxcycles=100)'
                else:
                    job_type_1 = 'opt=(calcfc, noeigentest)'
                job_type_2 = 'freq iop(7/33=1)'
                if self.fine:
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                    if self.is_ts:
                        job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5, maxcycles=100)'
                    else:
                        job_type_1 = 'opt=(calcfc, noeigentest, tight)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                self.input += """@@@

$molecule
   read
$end

$rem
   JOBTYPE       {job_type_2}
   METHOD        {method}
   UNRESTRICTED  {restricted}
   BASIS         {basis}
   SCF_GUESS     read
$end

"""
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                job_type_2 = 'freq'
                if self.fine:
                    fine = '\n   GEOM_OPT_TOL_GRADIENT 15\n   GEOM_OPT_TOL_DISPLACEMENT 60\n   GEOM_OPT_TOL_ENERGY 5'
            elif self.software == 'molpro':
                if self.is_ts:
                    job_type_1 = "\noptg,root=2,method=qsd,readhess,savexyz='geometry.xyz'"
                else:
                    job_type_1 = "\noptg,savexyz='geometry.xyz"
                job_type_2 = '\n{frequencies;\nthermo;\nprint,HESSIAN,thermo;}'
            elif self.software == 'orca':
                if self.is_ts:
                    job_type_1 = 'OptTS'
                    orca_options_blocks_dict['Calc_Hess'] = """
%geom
    Calc_Hess true # calculation of the exact Hessian before the first opt step
end               
"""
                else:
                    job_type_1 = 'Opt'
                job_type_2 = '!Freq'
                if self.fine:
                    user_fine_opt_convergence = ''
                    if self.args:
                        try:
                            user_fine_opt_convergence = self.args['keyword'].get('fine_opt_convergence', '')
                        except KeyError:
                            user_fine_opt_convergence = ''
                    fine_opt_convergence = user_fine_opt_convergence.lower() if user_fine_opt_convergence \
                        else orca_default_options_dict['opt']['keyword'].get('fine_opt_convergence', '').lower()
                    if not fine_opt_convergence:
                        raise InputError(
                            'Orca fine optimization convergence is not specified. Please specify this variable either '
                            'in the settings.py as default options or in the input file as additional options.')
                    orca_options_keywords_dict['fine_opt_convergence'] = fine_opt_convergence
                else:
                    user_opt_convergence = ''
                    if self.args:
                        try:
                            user_opt_convergence = self.args['keyword'].get('opt_convergence', '')
                        except KeyError:
                            user_opt_convergence = ''
                    opt_convergence = user_opt_convergence.lower() if user_opt_convergence \
                        else orca_default_options_dict['opt']['keyword'].get('opt_convergence', '').lower()
                    if not opt_convergence:
                        raise InputError(
                            'Orca optimization convergence is not specified. Please specify this variable either in '
                            'the settings.py as default options or in the input file as additional options.')
                    orca_options_keywords_dict['opt_convergence'] = opt_convergence
                use_num_freq = orca_default_options_dict['freq']['keyword'].get('use_num_freq', False)
                if self.args:
                    try:
                        use_num_freq = self.args['keyword'].get('use_num_freq', False)
                    except KeyError:
                        use_num_freq = False
                if use_num_freq:
                    orca_options_keywords_dict['freq_type'] = 'NumFreq'
                    logger.info(f'Using numeric frequencies calculation in Orca. Notice that this job will be '
                                f'very time consuming.')
                self.add_to_args(val=' '.join(orca_options_keywords_dict.values()))
                self.add_to_args(val='\n'.join(orca_options_blocks_dict.values()),
                                 key1='block')

        elif self.job_type == 'sp':
            if self.software == 'gaussian':
                job_type_1 = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                job_type_1 = 'sp'
            elif self.software == 'molpro':
                pass
            elif self.software == 'orca':
                job_type_1 = 'sp'
            elif self.software == 'terachem':
                job_type_1 = 'energy'

        elif self.job_type == 'composite':
            if self.software == 'gaussian':
                if self.fine:
                    fine = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                if self.is_ts:
                    job_type_1 = 'opt=(ts, calcfc, noeigentest, tight, maxstep=5, maxcycles=100)'
                else:
                    if self.level.method in ['rocbs-qb3']:
                        # No analytic 2nd derivatives (FC) for these methods
                        job_type_1 = 'opt=(noeigentest, tight)'
                    else:
                        job_type_1 = 'opt=(calcfc, noeigentest, tight)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=mix'
                else:
                    job_type_1 += ' guess=mix'
                if 'paraskevas' in self.level.method:
                    # convert cbs-qb3-paraskevas to cbs-qb3
                    self.level.method = 'cbs-qb3'
            else:
                raise JobError('Currently composite methods are only supported in gaussian')

        elif self.job_type == 'scan':
            if divmod(360, self.scan_res)[1]:
                raise JobError(f'Scan job got an illegal rotor scan resolution of {self.scan_res}')
            if self.scan is not None and self.scan:
                scans = [' '.join([str(num) for num in self.scan])]
            elif self.directed_scans is not None:
                scans = list()
                for directed_scan in self.directed_scans:
                    scans.append(' '.join([str(num) for num in directed_scan]))
            else:
                raise JobError(f'A scan job must either get a `scan` or a `directed_scans` argument.\n'
                               f'Got neither for job {self.job_name} of {self.species_label}.')
            if self.software == 'gaussian':
                ts = 'ts, ' if self.is_ts else ''
                job_type_1 = f'opt=({ts}modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) ' \
                             f'integral=(grid=ultrafine, Acc2E=12)'
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
                scan_string = ''
                for scan in scans:
                    scan_string += f'D {scan} S {int(360 / self.scan_res)} {self.scan_res:.1f}\n'
            elif self.software == 'qchem':
                if self.is_ts:
                    job_type_1 = 'ts'
                else:
                    job_type_1 = 'opt'
                dihedral1 = int(calculate_dihedral_angle(coords=self.xyz['coords'], torsion=self.scan))
                scan_string = '\n$scan\n'
                for scan in scans:
                    scan_string += f'tors {scan} {dihedral1} {dihedral1 + 360.0} {self.scan_res}\n'
                scan_string += '$end\n'
            elif self.software == 'terachem':
                if self.is_ts:
                    job_type_1 = 'ts\nnew_minimizer yes'
                else:
                    job_type_1 = 'minimize\nnew_minimizer yes'
                dihedral1 = int(calculate_dihedral_angle(coords=self.xyz['coords'], torsion=self.scan))
                scan_string = '\n$constraint_scan\n'
                for _ in scans:
                    scan_ = '_'.join([str(num) for num in self.scan])
                    num_points = int(360.0 / self.scan_res) + 1
                    scan_string += f'    dihedral {dihedral1} {dihedral1 + 360.0} {num_points} {scan_}\n'
                scan_string += '$end\n'
            else:
                raise JobError(f'Currently rotor scan is only supported in Gaussian, QChem, and TeraChem. Got:\n'
                               f'job type: {self.job_type}\n'
                               f'software: {self.software}\n'
                               f'level of theory: {self.level.method + "/" + self.level.basis}')

        elif self.job_type == 'directed_scan':
            # this is either a constrained opt job or an sp job (depends on self.directed_scan_type).
            # If opt, the dihedrals in self.directed_dihedral are constrained.
            if self.software == 'gaussian':
                if 'sp' in self.directed_scan_type:
                    job_type_1 = 'scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)'
                else:
                    ts = 'ts, ' if self.is_ts else ''
                    job_type_1 = f'opt=({ts}modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) ' \
                                 f'integral=(grid=ultrafine, Acc2E=12)'
                    for directed_scan, directed_dihedral in zip(self.directed_scans, self.directed_dihedrals):
                        scan_atoms = ' '.join([str(num) for num in directed_scan])
                        scan_string += 'D {scan} ={dihedral} B\nD {scan} F\n'.format(
                            scan=scan_atoms, dihedral='{0:.2f}'.format(directed_dihedral))
                if self.checkfile is not None:
                    job_type_1 += ' guess=read'
                else:
                    job_type_1 += ' guess=mix'
            elif self.software == 'qchem':
                # following https://manual.q-chem.com/5.2/Ch10.S3.SS4.html
                if 'sp' in self.directed_scan_type:
                    job_type_1 = 'sp'
                else:
                    if self.is_ts:
                        job_type_1 = 'ts'
                    else:
                        job_type_1 = 'opt'
                    constraint = '\n    CONSTRAINT\n'
                    for directed_scan, directed_dihedral in zip(self.directed_scans, self.directed_dihedrals):
                        scan_atoms = ' '.join([str(num) for num in directed_scan])
                        constraint += '        tors {scan} {dihedral}\n'.format(
                            scan=scan_atoms, dihedral='{0:.2f}'.format(directed_dihedral))
                    constraint += '    ENDCONSTRAINT\n'
            else:
                raise ValueError(f'Currently directed rotor scans are only supported in Gaussian and QChem. '
                                 f'Got: {self.software} using the {self.level} level of theory')

        if self.software == 'gaussian':
            if self.level.method[:2] == 'ro':
                self.add_to_args(val='use=L506')
            else:
                # xqc will do qc (quadratic convergence) if the job fails w/o it, so use by default
                self.add_to_args(val='scf=xqc')

        if self.software == 'terachem':
            # TeraChem requires an additional xyz file.
            # Note: the xyz filename must correspond to the xyz filename specified in TeraChem's input file!
            save_geo(xyz=self.xyz, path=self.local_path, filename='coord', format_='xyz')

        if self.job_type == 'irc':
            if self.irc_direction is None or self.irc_direction not in ['forward', 'reverse']:
                raise JobError(f'The IRC direction must be either "forward" or "reverse", got {self.irc_direction}.')
            if self.fine:
                # Note that the Acc2E argument is not available in Gaussian03
                fine = 'scf=(direct) integral=(grid=ultrafine, Acc2E=12)'
            job_type_1 = f'irc=(CalcAll,{self.irc_direction},maxpoints=50,stepsize=7)'
            if self.checkfile is not None:
                job_type_1 += ' guess=read'
            else:
                job_type_1 += ' guess=mix'

        elif self.job_type == 'gsm':  # TODO
            pass

        if self.software == 'gaussian' and self.level.solvation_method is not None:
            job_type_1 += f' SCRF=({self.level.solvation_method},Solvent={self.level.solvent})'

        if 'mrci' in self.level.method:
            if self.software != 'molpro':
                raise JobError('Can only run MRCI on Molpro, not {0}'.format(self.software))
            if self.occ > 16:
                raise JobError('Will not execute an MRCI calculation with more than 16 occupied orbitals.'
                               'Selective occ, closed, core, frozen keyword still not implemented.')
            else:
                try:
                    self.input = input_files['mrci'].format(memory=self.input_file_memory, xyz=xyz_to_str(self.xyz),
                                                            basis=self.level.basis, spin=self.spin, charge=self.charge,
                                                            args=self.args['keyword'])
                except KeyError:
                    logger.error('Could not interpret all input file keys in\n{0}'.format(self.input))
                    raise
        else:
            job_options_blocks = '\n\n'.join(self.args['block'].values()) \
                if 'block' in self.args and self.args['block'] else ''
            job_options_keywords = ' '.join(self.args['keyword'].values()) \
                if 'keyword' in self.args and self.args['keyword'] else ''
            try:
                self.input = self.input.format(memory=int(self.input_file_memory),
                                               method=self.level.method,
                                               slash=slash,
                                               slash_2=slash_2,
                                               basis=self.level.basis or '',
                                               charge=self.charge,
                                               cpus=self.cpu_cores,
                                               multiplicity=self.multiplicity,
                                               spin=self.spin,
                                               xyz=xyz_to_str(self.xyz),
                                               job_type_1=job_type_1,
                                               job_type_2=job_type_2,
                                               scan=scan_string,
                                               restricted=restricted,
                                               fine=fine,
                                               shift=self.shift,
                                               scan_trsh=self.scan_trsh,
                                               bath=self.bath_gas,
                                               constraint=constraint,
                                               job_options_blocks=job_options_blocks,
                                               job_options_keywords=job_options_keywords,
                                               method_class=method_class,
                                               auxiliary_basis=self.level.auxiliary_basis or '',
                                               dispersion=self.dispersion,
                                               ) \
                    if self.input is not None else None
            except KeyError as e:
                logger.error(f'Could not interpret all input file keys in\n{self.input}\nGot:\n{e}')
                raise

        if not self.testing:
            if not os.path.exists(self.local_path):
                os.makedirs(self.local_path)
            if self.input is not None:
                with open(os.path.join(self.local_path, input_filenames[self.software]), 'w') as f:
                    f.write(self.input)
            if self.server != 'local':
                self._upload_input_file()
            else:
                self.initial_time = get_last_modified_time(
                    file_path=os.path.join(self.local_path, submit_filenames[servers[self.server]['cluster_soft']]))
                # copy additional input files to local running directory
                for up_file in self.additional_files_to_upload:
                    if up_file['source'] == 'path':
                        source_path = up_file['local']
                        destination_path = os.path.join(self.local_path, up_file['name'])
                        shutil.copyfile(source_path, destination_path)
                    elif up_file['source'] == 'input_files':
                        with open(os.path.join(self.local_path, up_file['name']), 'w') as f:
                            f.write(input_files[up_file['local']])

            if self.checkfile is not None and os.path.isfile(self.checkfile):
                self._upload_check_file(local_check_file_path=self.checkfile)

    def _download_output_file(self):
        """
        Download ESS output, orbitals check file, and the Gaussian check file, if relevant.
        """
        with SSHClient(self.server) as ssh:

            # download output file
            remote_file_path = os.path.join(self.remote_path, output_filenames[self.software])
            ssh.download_file(remote_file_path=remote_file_path, local_file_path=self.local_path_to_output_file)
            if not os.path.isfile(self.local_path_to_output_file):
                raise JobError(f'output file for {self.job_name} was not downloaded properly')
            self.final_time = ssh.get_last_modified_time(remote_file_path=remote_file_path)

            # download orbitals FChk file
            if self.job_type == 'orbitals':
                remote_file_path = os.path.join(self.remote_path, 'input.FChk')
                ssh.download_file(remote_file_path=remote_file_path, local_file_path=self.local_path_to_orbitals_file)
                if not os.path.isfile(self.local_path_to_orbitals_file):
                    logger.warning(f'Orbitals FChk file for {self.job_name} was not downloaded properly '
                                   f'(this is not the Gaussian formatted check file...)')

            # download Gaussian check file
            if self.software.lower() == 'gaussian':
                remote_check_file_path = os.path.join(self.remote_path, 'check.chk')
                ssh.download_file(remote_file_path=remote_check_file_path, local_file_path=self.local_path_to_check_file)
                if not os.path.isfile(self.local_path_to_check_file):
                    logger.warning(f'Gaussian check file for {self.job_name} was not downloaded properly')

            # download Orca .hess hessian file generated by frequency calculations
            # Hessian is useful when the user would like to project rotors
            if self.software.lower() == 'orca' and self.job_type == 'freq':
                remote_hess_file_path = os.path.join(self.remote_path, 'input.hess')
                ssh.download_file(remote_file_path=remote_hess_file_path, local_file_path=self.local_path_to_hess_file)
                if not os.path.isfile(self.local_path_to_hess_file):
                    logger.warning(f'Orca hessian file for {self.job_name} was not downloaded properly')

            # download Lennard_Jones data file
            if self.software.lower() == 'onedmin':
                remote_lj_file_path = os.path.join(self.remote_path, 'lj.dat')
                ssh.download_file(remote_file_path=remote_lj_file_path, local_file_path=self.local_path_to_lj_file)
                if not os.path.isfile(self.local_path_to_lj_file):
                    logger.warning(f'Lennard-Jones data file for {self.job_name} was not downloaded properly')

            # download molpro log file (in addition to the output file)
            if self.software.lower() == 'molpro':
                remote_log_file_path = os.path.join(self.remote_path, output_filenames[self.software])
                ssh.download_file(remote_file_path=remote_log_file_path, local_file_path=self.local_path_to_output_file)
                if not os.path.isfile(self.local_path_to_output_file):
                    logger.warning(f'Could not download Molpro log file for {self.job_name} '
                                   f'(this is not the output file)')

            # download terachem files (in addition to the output file)
            if self.software.lower() == 'terachem':
                base_path = os.path.join(self.remote_path, 'scr')
                filenames = ['results.dat', 'output.molden', 'charge.xls', 'charge_mull.xls', 'optlog.xls', 'optim.xyz',
                             'Frequencies.dat', 'I_matrix.dat', 'Mass.weighted.modes.dat', 'moments_of_inertia.dat',
                             'output.basis', 'output.geometry', 'output.molden', 'Reduced.mass.dat', 'results.dat']
                for filename in filenames:
                    remote_log_file_path = os.path.join(base_path, filename)
                    local_log_file_path = os.path.join(self.local_path, 'scr', filename)
                    ssh.download_file(remote_file_path=remote_log_file_path, local_file_path=local_log_file_path)
                xyz_path = os.path.join(base_path, 'optim.xyz')
                if os.path.isfile(xyz_path):
                    self.local_path_to_xyz = xyz_path

    def run(self):
        """
        Execute the Job.
        """
        if self.fine:
            logger.info(f'Running job {self.job_name} for {self.species_label} (fine opt)')
        elif self.directed_dihedrals is not None and self.directed_scans is not None:
            dihedrals = ['{0:.2f}'.format(dihedral) for dihedral in self.directed_dihedrals]
            logger.info(f'Running job {self.job_name} for {self.species_label} (pivots: {self.directed_scans}, '
                        f'dihedrals: {dihedrals})')
        elif self.pivots:
            logger.info(f'Running job {self.job_name} for {self.species_label} (pivots: {self.pivots})')
        else:
            logger.info(f'Running job {self.job_name} for {self.species_label}')
        logger.debug('writing submit script...')
        self.write_submit_script()
        logger.debug('writing input file...')
        self.write_input_file()
        if self.server != 'local':
            logger.debug('submitting job...')
            # submit_job returns job server status and job server id
            with SSHClient(self.server) as ssh:
                self.job_status[0], self.job_id = ssh.submit_job(remote_path=self.remote_path)
        else:
            # running locally
            self.job_status[0], self.job_id = submit_job(path=self.local_path)

    def troubleshoot_server(self):
        """
        Troubleshoot server errors.
        """
        node, run_job = trsh_job_on_server(server=self.server, job_name=self.job_name, job_id=self.job_id,
                                           job_server_status=self.job_status[0], remote_path=self.remote_path,
                                           server_nodes=self.server_nodes)
        if node is not None:
            self.server_nodes.append(node)
        if run_job:
            # resubmit job
            self.run()

    def deduce_software(self):  ## Todo: relocate to where the JobAdapter needs to be determined
        """
        Deduce the software to be used.

        Returns: str
            The deduced software.
        """
        self.level.deduce_software(job_type=self.job_type)
        if self.level.software is not None:
            self.software = self.level.software
        else:
            logger.error(f'Could not determine software for job {self.job_name}')
            logger.error(f'job_num: {self.job_num}')
            logger.error(f'job_type: {self.job_type}')
            logger.error(f'level_of_theory: {self.level}')
            logger.error(f'ess_trsh_methods: {self.ess_trsh_methods}')
            logger.error(f'args: {pformat(self.args)}')
            available_ess = list(self.ess_settings.keys())
            if 'gaussian' in available_ess:
                logger.error('Setting it to Gaussian')
                self.level.software = 'gaussian'
            elif 'qchem' in available_ess:
                logger.error('Setting it to QChem')
                self.level.software = 'qchem'
            elif 'orca' in available_ess:
                logger.error('Setting it to Orca')
                self.level.software = 'orca'
            elif 'molpro' in available_ess:
                logger.error('available_ess it to Molpro')
                self.level.software = 'molpro'
            elif 'terachem' in available_ess:
                logger.error('Setting it to TeraChem')
                self.level.software = 'terachem'
