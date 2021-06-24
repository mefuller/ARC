"""
A module for checking the quality of TS-related calculations, contains helper functions for Scheduler.
"""

import logging
import os

import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from rmgpy.exceptions import ActionError

import arc.rmgdb as rmgdb
from arc import parser
from arc.common import (ARC_PATH,
                        convert_list_index_0_to_1,
                        extremum_list,
                        get_logger,
                        read_yaml_file,
                        )

if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from arc.job.adapter import JobAdapter
    from arc.species.species import ARCSpecies, TSGuess
    from arc.reaction import ARCReaction

logger = get_logger()


def check_ts(reaction: 'ARCReaction',
             verbose: bool = True,
             parameter: str = 'E0',
             rxn_zone_atom_indices: Optional[List[int]] = None,
             ):
    """
    Check the TS in terms of energy, normal mode displacement, and IRC.
    Populates the ``TS.ts_checks`` dictionary.
    Note that the 'freq' check is done in Scheduler.check_negative_freq() and not here.

    Args:
        reaction ('ARCReaction'): THe reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
        parameter (str, optional): The energy parameter to consider ('E0' or 'e_elect').
        rxn_zone_atom_indices (List[int], optional): The 0-indexed atom indices of atoms participating in the
                                                     reaction (which form the reactive zone of the TS).
    """
    if not reaction.ts_species.ts_checks['E0'] and not reaction.ts_species.ts_checks['e_elect']:
        check_ts_energy(reaction=reaction, verbose=verbose, parameter=parameter)
    if not reaction.ts_species.ts_checks['normal_mode_displacement'] and rxn_zone_atom_indices is not None:
        check_normal_mode_displacement(reaction, rxn_zone_atom_indices)
    # if not self.ts_species.ts_checks['IRC'] and IRC_wells is not None:
    #     self.check_irc(rxn_zone_atom_indices)


def check_ts_energy(reaction: 'ARCReaction',
                    verbose: bool = True,
                    parameter: str = 'E0',
                    ) -> None:
    """
    Check that the TS E0 or electronic energy is above both reactant and product wells.
    By default E0 is checked first, if not available for all species and TS, the electronic energy is checked.
    Sets the respective energy parameter in the ``TS.ts_checks`` dictionary.

    Args:
        reaction ('ARCReaction'): THe reaction for which the TS is checked.
        verbose (bool, optional): Whether to print logging messages.
        parameter (str, optional): The energy parameter to consider ('E0' or 'e_elect').
    """
    if parameter not in ['E0', 'e_elect']:
        raise ValueError(f"The energy parameter must be either 'E0' or 'e_elect', got: {parameter}")

    # Determine E0 and e_elect.
    r_e0 = None if any([spc.e0 is None for spc in reaction.r_species]) \
        else sum(spc.e0 * reaction.get_species_count(species=spc, well=0) for spc in reaction.r_species)
    p_e0 = None if any([spc.e0 is None for spc in reaction.p_species]) \
        else sum(spc.e0 * reaction.get_species_count(species=spc, well=1) for spc in reaction.p_species)
    ts_e0 = reaction.ts_species.e0
    r_e_elect = None if any([spc.e_elect is None for spc in reaction.r_species]) \
        else sum(spc.e_elect * reaction.get_species_count(species=spc, well=0) for spc in reaction.r_species)
    p_e_elect = None if any([spc.e_elect is None for spc in reaction.p_species]) \
        else sum(spc.e_elect * reaction.get_species_count(species=spc, well=1) for spc in reaction.p_species)
    ts_e_elect = reaction.ts_species.e_elect

    # Determine the parameter by which to compare.
    r_e = r_e0 if parameter == 'E0' else r_e_elect
    p_e = p_e0 if parameter == 'E0' else p_e_elect
    ts_e = ts_e0 if parameter == 'E0' else ts_e_elect
    min_e = extremum_list([r_e, p_e, ts_e], return_min=True)
    e_str = 'E0' if parameter == 'E0' else 'electronic energy'

    if any([val is not None for val in [r_e, p_e, ts_e]]):
        if verbose:
            r_text = f'{r_e - min_e:.2f} kJ/mol' if r_e is not None else 'None'
            ts_text = f'{ts_e - min_e:.2f} kJ/mol' if ts_e is not None else 'None'
            p_text = f'{p_e - min_e:.2f} kJ/mol' if p_e is not None else 'None'
            logger.info(f'\nReaction {reaction.label} has the following path {e_str}:\n'
                        f'Reactants: {r_text}\n'
                        f'TS: {ts_text}\n'
                        f'Products: {p_text}')

        if all([val is not None for val in [r_e, p_e, ts_e]]):
            # We have all params, we can make a quantitative decision.
            if ts_e > r_e and ts_e > p_e:
                # TS is above both wells.
                reaction.ts_species.ts_checks[parameter] = True
                return
            # TS is not above both wells.
            if verbose:
                logger.error(f'TS of reaction {reaction.label} has a lower {e_str} value than expected.')
                reaction.ts_species.ts_checks[parameter] = False
                return
        # We don't have all params (some are ``None``).
    # We don't have any params (they are all ``None``), or we don't have any params and were only checking E0.
    if parameter == 'E0':
        # Use e_elect instead:
        logger.debug(f'Could not get all E0 values for reaction {reaction.label}, comparing energies using e_elect.')
        check_ts_energy(reaction=reaction, verbose=verbose, parameter='e_elect')
        return
    if verbose:
        logger.info('\n')
        logger.error(f"Could not get {e_str} of all species in reaction {reaction.label}. Cannot check TS.\n")
    # We don't really know, assume ``True``
    reaction.ts_species.ts_checks[parameter] = True
    reaction.ts_species.ts_checks['warnings'] += 'Could not determine TS energy relative to the wells; '


def check_normal_mode_displacement(reaction: 'ARCReaction',
                                   rxn_zone_atom_indices: List[int]):
    """
    Check the normal mode displacement by making sure that the atom indices derived from the major motion
    (the major normal mode displacement) fit the expected RMG reaction template.
    Note that RMG does not differentiate well between hydrogen atoms since it thinks in 2D.

    Args:
        reaction ('ARCReaction'): THe reaction for which the TS is checked.
        rxn_zone_atom_indices (List[int], optional): The 0-indexed indices of atoms participating in the
                                                     reaction (which form the reactive zone of the TS).
    """
    # todo: symmetry like in C[CH]C will give different results. need to consider degeneracy in RMG and check if one path fits
    # todo: hydrogens are problematic, can't know using 2D which H on a CH3 migrated
    rmg_rxn = reaction.rmg_reaction.copy()
    try:
        reaction.family.add_atom_labels_for_reaction(reaction=rmg_rxn, output_with_resonance=False, save_order=True)
    except (ActionError, ValueError):
        reaction.ts_species.ts_checks['normal_mode_displacement'] = True
        reaction.ts_species.ts_checks['warnings'] += 'Could not determine atom labels from RMG, ' \
                                                     'cannot check normal mode displacement; '
    else:
        r_labels, p_labels = list(), list()
        for reactant in rmg_rxn.reactants:
            print([(i, atom.label) for i, atom in enumerate(reactant.atoms)])
            r_labels.extend([i for i, atom in enumerate(reactant.atoms) if atom.label])
        for product in rmg_rxn.products:
            print([(i, atom.label) for i, atom in enumerate(product.atoms)])
            p_labels.extend([i for i, atom in enumerate(product.atoms) if atom.label])
        r_labels, p_labels = list(set(r_labels)), list(set(p_labels))
        r_labels.sort()
        p_labels.sort()
        rxn_zone_atom_indices.sort()
        reaction.ts_species.rxn_zone_atom_indices = rxn_zone_atom_indices
        if r_labels == p_labels == rxn_zone_atom_indices:
            reaction.ts_species.ts_checks['normal_mode_displacement'] = True


def check_ts_freq_job(species: 'ARCSpecies',
                      reaction: 'ARCReaction',
                      job: 'JobAdapter',
                      ) -> bool:
    """
    Parse normal mode displacement. Invalidate rotors that break a TS,
    and determine whether the the normal mode displacement make sense for the given RMG family.

    Args:
        species ('ARCSpecies'): The TS species.
        reaction ('ARCReaction'): The respective reaction object instance.
        job ('JobAdapter'): The frequency job object instance.

    Returns:
        bool: Whether a new TS guess should be tried.
    """
    switch_ts = False
    if not species.is_ts:
        return switch_ts
    freqs, normal_mode_disp = parser.parse_normal_mode_displacement(path=job.local_path_to_output_file,
                                                                    raise_error=False)
    rxn_zone_atom_indices = get_indices_of_atoms_participating_in_reaction(normal_mode_disp=normal_mode_disp,
                                                                           freqs=freqs,
                                                                           ts_guesses=species.ts_guesses,
                                                                           )
    invalidate_rotors_with_both_pivots_in_a_reactive_zone(species=species,
                                                          rxn_zone_atom_indices=rxn_zone_atom_indices)

    # Check the normal mode displacement.
    check_ts(reaction=reaction, verbose=False, rxn_zone_atom_indices=rxn_zone_atom_indices)
    if not species.ts_checks['normal_mode_displacement']:
        logger.warning(f'The computed normal displacement mode of TS {species.label} ({rxn_zone_atom_indices}) '
                       f'does not match the expected labels from RMG '
                       f'({species.rxn_zone_atom_indices}). Switching TS conformer.')
        switch_ts = True
    return switch_ts


def invalidate_rotors_with_both_pivots_in_a_reactive_zone(species: 'ARCSpecies',
                                                          rxn_zone_atom_indices: List[int],
                                                          ):
    """
    Invalidate rotors in which both pivots are included in the reactive zone.

    Args:
        species ('ARCSpecies'): The TS species.
        rxn_zone_atom_indices (List[int]): The 0-indexed indices of atoms participating in the reaction zone of a TS.
    """
    if not species.rotors_dict:
        species.determine_rotors()
    rxn_zone_atom_indices_1 = convert_list_index_0_to_1(rxn_zone_atom_indices)
    for key, rotor in species.rotors_dict.items():
        if rotor['pivots'][0] in rxn_zone_atom_indices_1 and rotor['pivots'][1] in rxn_zone_atom_indices_1:
            rotor['success'] = False
            if 'pivTS' not in rotor['invalidation_reason']:
                rotor['invalidation_reason'] += 'Pivots participate in the TS reaction zone (code: pivTS). '
                logging.info(f"\nNot considering rotor {key} with pivots {rotor['pivots']} in TS {species.label}\n")


def get_indices_of_atoms_participating_in_reaction(normal_mode_disp: np.ndarray,
                                                   freqs: np.ndarray,
                                                   ts_guesses: List['TSGuess'],
                                                   ) -> List[int]:
    """
    Get the indices of the atoms participating in the reaction (which form the reactive zone of the TS).

    Args:
        normal_mode_disp (np.ndarray): The normal displacement modes array.
        freqs (np.ndarray): Entries are frequency values.
        ts_guesses (List[TSGuess]): The TSGuess object instances of the TS Species.

    Returns:
        List[int]: The indices of the atoms participating in the reaction.
                   The indices are 0-indexed and sorted in an increasing order.
    """
    normal_disp_mode_rms = get_rms_from_normal_mode_disp(normal_mode_disp, freqs)
    num_of_atoms = get_expected_num_atoms_with_largest_normal_mode_disp(
        normal_disp_mode_rms=normal_disp_mode_rms,
        ts_guesses=ts_guesses,
    )
    return sorted(range(len(normal_disp_mode_rms)), key=lambda i: normal_disp_mode_rms[i], reverse=True)[:num_of_atoms]


def get_rms_from_normal_mode_disp(normal_mode_disp: np.ndarray,
                                  freqs: np.ndarray,
                                  ) -> List[float]:
    """
    Get the root mean squares of the normal displacement modes.

    Args:
        normal_mode_disp (np.ndarray): The normal displacement modes array.
        freqs (np.ndarray): Entries are frequency values.

    Returns:
        List[float]: The RMS of the normal displacement modes.
    """
    rms = list()
    mode_index = get_index_of_abs_largest_neg_freq(freqs)
    nmd = normal_mode_disp[mode_index]
    for entry in nmd:
        rms.append((entry[0] ** 2 + entry[1] ** 2 + entry[2] ** 2) ** 0.5)
    return rms


def get_index_of_abs_largest_neg_freq(freqs: np.ndarray) -> Optional[int]:
    """
    Get the index of the |largest| negative frequency.

    Args:
        freqs (np.ndarray): Entries are frequency values.

    Returns:
        Optional[int]: The 0-index of the largest absolute negative frequency.
    """
    if not len(freqs) or all(freq > 0 for freq in freqs):
        return None
    return list(freqs).index(min(freqs))


def get_expected_num_atoms_with_largest_normal_mode_disp(normal_disp_mode_rms: List[float],
                                                         ts_guesses: List['TSGuess'],
                                                         ) -> int:
    """
    Get the number of atoms that are expected to have the largest normal mode displacement for the TS
    (considering all families). This is a wrapper for ``get_rxn_normal_mode_disp_atom_number()``.
    It is theoretically possible that TSGuesses of the same species will belong to different families.

    Args:
        normal_disp_mode_rms (List[float]): The RMS of the normal displacement modes..
        ts_guesses (List['TSGuess']): The TSGuess objects of a TS species.

    Returns:
        int: The number of atoms to consider that have a significant motions in the normal mode displacement.
    """
    families = list(set([tsg.family for tsg in ts_guesses]))
    num_of_atoms = max([get_rxn_normal_mode_disp_atom_number(rxn_family=family, rms_list=normal_disp_mode_rms)
                        for family in families])
    return num_of_atoms


def get_rxn_normal_mode_disp_atom_number(rxn_family: str,
                                         rms_list: Optional[List[float]] = None,
                                         ) -> int:
    """
    Get the number of atoms expected to have the largest normal mode displacement per family.
    If ``rms_list`` is given, also include atoms with an rms value close to the lowest rms still considered.

    Args:
        rxn_family (str): The reaction family label.
        rms_list (List[float], optional): The root mean squares of the normal mode displacements.

    Raises:
        TypeError: If ``rms_list`` is not ``None`` and is either not a list or does not contain floats.

    Returns:
        int: The respective number of atoms.
    """
    if not isinstance(rxn_family, str):
        raise TypeError(f'rxn_family must be a string, got {rxn_family} which is a {type(rxn_family)}.')
    if rms_list is not None and (
            not isinstance(rms_list, list) or not all(isinstance(entry, float) for entry in rms_list)):
        raise TypeError(f'rms_list must be a non empty list, got {rms_list} of type {type(rms_list)}.')
    content = read_yaml_file(os.path.join(ARC_PATH, 'data', 'rxn_normal_mode_disp.yml'))
    number_by_family = content.get(rxn_family, 3)
    if rms_list is None or not len(rms_list):
        return number_by_family
    entry = None
    rms_list = rms_list.copy()
    for i in range(number_by_family):
        entry = max(rms_list)
        rms_list.pop(rms_list.index(entry))
    if entry is not None:
        for rms in rms_list:
            if (entry - rms) / entry < 0.12:
                number_by_family += 1
    return number_by_family


def find_equivalent_atoms_in_reactants(reaction: 'ARCReaction') -> Optional[List[List[int]]]:
    """
    Find atom indices that are equivalent in the reactants of an ARCReaction
    in the sense that they represent degenerate reaction sites that are indifferentiable in 2D.
    Bridges between RMG reaction templates and ARC's 3D TS structures.
    Running indices in the returned structure relate to reactant_0 + reactant_1 + ...

    Args:
        reaction ('ARCReaction'): An ARCReaction object instance.

    Returns:
        Optional[List[List[int]]]: Entries are lists of 0-indices, each such list represents equivalent atoms.
    """
    if reaction.family is None:
        db = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(db)
        reaction.determine_family(db)
    if reaction.family is None:
        return None
    rmg_reactions = reaction.family.generate_reactions(reactants=[spc.mol for spc in reaction.r_species],
                                                       products=[spc.mol for spc in reaction.p_species],
                                                       prod_resonance=True,
                                                       delete_labels=False,
                                                       )
    dicts = [get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reaction) for rmg_reaction in rmg_reactions]
    equivalence_map = dict()
    for index_dict in dicts:
        for key, value in index_dict.items():
            if key in equivalence_map:
                equivalence_map[key].append(value)
            else:
                equivalence_map[key] = [value]
    equivalent_indices = list(list(set(equivalent_list)) for equivalent_list in equivalence_map.values())
    return equivalent_indices


def get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(reaction: 'TemplateReaction') -> Optional[Dict[str, int]]:
    """
    Get the RMG reaction labels and the corresponding 0-indexed atom indices
    for all labeled atoms in an TemplateReaction.

    Args:
        reaction ('TemplateReaction'): An RMG family TemplateReaction object instance.

    Returns:
        Optional[Dict[str, int]]: Keys are labels (e.g., '*1'), values are corresponding 0-indices atoms.
    """
    if not hasattr(reaction, 'labeledAtoms') or not len(reaction.labeledAtoms):
        return None
    index_dict = dict()
    reactant_atoms, product_atoms = list(), list()
    for mol in reaction.reactants:
        reactant_atoms.extend([atom for atom in mol.atoms])
    for mol in reaction.products:
        product_atoms.extend([atom for atom in mol.atoms])
    for labeled_atom in reaction.labeledAtoms:
        for i, atom in enumerate(reactant_atoms):
            if atom.id == labeled_atom[1].id:
                index_dict[labeled_atom[0]] = i
                break
    return index_dict
