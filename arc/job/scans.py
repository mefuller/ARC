"""
A module for preparing data structures for directed scans
"""

from typing import TYPE_CHECKING

from arc.common import get_logger

if TYPE_CHECKING:
    from arc.species import ARCSpecies

logger = get_logger()


def generate_scan_points(species: 'ARCSpecies',
                         scan_res: float,
                         ) -> dict:
    """
    Generate all coordinates in advance for brute_force (non-continuous) directed scans.
    Directed scan types could be one of the following: ``'brute_force_sp'``, ``'brute_force_opt'``,
    ``'brute_force_sp_diagonal'``, or ``'brute_force_opt_diagonal'``.
    The differentiation between ``'sp'`` and ``'opt'`` is done in at the Job level.

    Args:
        species (ARCSpecies): The species to consider.
        scan_res (float): THe scan resolution to use.

    Raises:
        ValueError: If the species directed scan type has an unexpected value.

    Returns: dict
        The data structures to be saved to the HDF5 file for this species.
    """
    species_data = dict()

    data.append({'label': species.label,
                 'status': 0,
                 'charge': species.charge,
                 'multiplicity': species.multiplicity,
                 'xyz1': conformer,
                 'xyz2': None,
                 'job_types': 'opt',
                 'level': self.level.as_dict(),
                 'constraints': self.constraints,
                 'execution_type': 'incore',
                 'xyz_out': None,
                 'electronic_energy': None,
                 'frequencies': None,
                 'error': None,
                 })


    for rotor_dict in species.rotors_dict.values():
        scans = rotor_dict['scan']
        pivots = rotor_dict['pivots']
        directed_scan_type = rotor_dict['directed_scan_type']
        xyz = species.get_xyz(generate=True)

        if 'cont' not in directed_scan_type and 'brute' not in directed_scan_type and 'ess' not in directed_scan_type:
            raise InputError(f'directed_scan_type must be either continuous or brute force, got: {directed_scan_type}')

        if 'ess' in directed_scan_type:
            # allow the ESS to control the scan
            self.run_job(label=label, xyz=xyz, level_of_theory=self.scan_level, job_type='scan',
                         directed_scan_type=directed_scan_type, directed_scans=scans, rotor_index=rotor_index,
                         pivots=pivots)

        elif 'brute' in directed_scan_type:
            # spawn jobs all at once
            dihedrals = dict()
            for scan in scans:
                original_dihedral = calculate_dihedral_angle(coords=xyz['coords'], torsion=scan)
                dihedrals[tuple(scan)] = [round(original_dihedral + i * scan_res
                                                if original_dihedral + i * scan_res <= 180.0
                                                else original_dihedral + i * scan_res - 360.0, 2)
                                          for i in range(int(360 / scan_res) + 1)]
            modified_xyz = xyz
            if 'diagonal' not in directed_scan_type:
                # increment dihedrals one by one (resulting in an ND scan)
                all_dihedral_combinations = list(itertools.product(*[dihedrals[tuple(scan)] for scan in scans]))
                for dihedral_tuple in all_dihedral_combinations:
                    for scan, dihedral in zip(scans, dihedral_tuple):
                        species.set_dihedral(scan=scan, deg_abs=dihedral, count=False,
                                                              xyz=modified_xyz)
                        modified_xyz = species.initial_xyz
                    rotor_dict['number_of_running_jobs'] += 1
                    self.run_job(label=label,
                                 xyz=modified_xyz,
                                 level_of_theory=self.scan_level,
                                 job_type='directed_scan',
                                 directed_scan_type=directed_scan_type,
                                 directed_scans=scans,
                                 directed_dihedrals=list(dihedral_tuple),
                                 rotor_index=rotor_index,
                                 pivots=pivots)
            else:
                # increment all dihedrals at once (resulting in a unique 1D scan along several changing dimensions)
                for i in range(len(dihedrals[tuple(scans[0])])):
                    for scan in scans:
                        dihedral = dihedrals[tuple(scan)][i]
                        species.set_dihedral(scan=scan, deg_abs=dihedral, count=False,
                                                              xyz=modified_xyz)
                        modified_xyz = species.initial_xyz
                    directed_dihedrals = [dihedrals[tuple(scan)][i] for scan in scans]
                    rotor_dict['number_of_running_jobs'] += 1
                    self.run_job(label=label,
                                 xyz=modified_xyz,
                                 level_of_theory=self.scan_level,
                                 job_type='directed_scan',
                                 directed_scan_type=directed_scan_type,
                                 directed_scans=scans,
                                 directed_dihedrals=directed_dihedrals,
                                 rotor_index=rotor_index,
                                 pivots=pivots,
                                 )

        elif 'cont' in directed_scan_type:
            # spawn jobs one by one
            if not len(rotor_dict['cont_indices']):
                rotor_dict['cont_indices'] = [0] * len(scans)
            if not len(rotor_dict['original_dihedrals']):
                rotor_dict['original_dihedrals'] = \
                    [f'{calculate_dihedral_angle(coords=xyz["coords"], torsion=scan, index=1):.2f}'
                     for scan in rotor_dict['scan']]  # stores as str for YAML
            scans = rotor_dict['scan']
            pivots = rotor_dict['pivots']
            max_num = 360 / scan_res + 1  # dihedral angles per scan
            original_dihedrals = list()
            for dihedral in rotor_dict['original_dihedrals']:
                f_dihedral = float(dihedral)
                original_dihedrals.append(f_dihedral if f_dihedral < 180.0 else f_dihedral - 360.0)
            if not any(rotor_dict['cont_indices']):
                # this is the first call for this cont_opt directed rotor, spawn the first job w/o changing dihedrals
                self.run_job(label=label,
                             xyz=species.final_xyz,
                             level_of_theory=self.scan_level,
                             job_type='directed_scan',
                             directed_scan_type=directed_scan_type,
                             directed_scans=scans,
                             directed_dihedrals=original_dihedrals,
                             rotor_index=rotor_index, pivots=pivots,
                             )
                rotor_dict['cont_indices'][0] += 1
                return
            else:
                # this is NOT the first call for this cont_opt directed rotor, check that ``xyz`` was given.
                if xyz is None:
                    # xyz is None only at the first time cont opt is spawned, where cont_index is [0, 0,... 0].
                    raise InputError('xyz argument must be given for a continuous scan job')
                # check whether this rotor is done
                if rotor_dict['cont_indices'][-1] == max_num - 1:  # 0-indexed
                    # no more counters to increment, all done!
                    logger.info(f'Completed all jobs for the continuous directed rotor scan for species {label} '
                                f'between pivots {pivots}')
                    self.process_directed_scans(label, pivots)
                    return

            modified_xyz = xyz
            dihedrals = list()
            for index, (original_dihedral, scan_) in enumerate(zip(original_dihedrals, scans)):
                dihedral = original_dihedral + rotor_dict['cont_indices'][index] * scan_res
                # change the original dihedral so we won't end up with two calcs for 180.0, but none for -180.0
                # (it only matters for plotting, the geometry is of course the same)
                dihedral = dihedral if dihedral <= 180.0 else dihedral - 360.0
                dihedrals.append(dihedral)
                # Only change the dihedrals in the xyz if this torsion corresponds to the current index,
                # or if this is a diagonal scan.
                # Species.set_dihedral() uses .final_xyz or the given xyz to modify the .initial_xyz
                # attribute to the desired dihedral.
                species.set_dihedral(scan=scan_, deg_abs=dihedral, count=False, xyz=modified_xyz)
                modified_xyz = species.initial_xyz
            self.run_job(label=label,
                         xyz=modified_xyz,
                         level_of_theory=self.scan_level,
                         job_type='directed_scan',
                         directed_scan_type=directed_scan_type,
                         directed_scans=scans,
                         directed_dihedrals=dihedrals,
                         rotor_index=rotor_index,
                         pivots=pivots,
                         )

            if 'diagonal' in directed_scan_type:
                # increment ALL counters for a diagonal scan
                rotor_dict['cont_indices'] = [rotor_dict['cont_indices'][0] + 1] * len(scans)
            else:
                # increment the counter sequentially (non-diagonal scan)
                for index in range(len(scans)):
                    if rotor_dict['cont_indices'][index] < max_num - 1:
                        rotor_dict['cont_indices'][index] += 1
                        break
                    elif (rotor_dict['cont_indices'][index] == max_num - 1 and index < len(scans) - 1):
                        rotor_dict['cont_indices'][index] = 0

    return species_data


