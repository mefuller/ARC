"""
A module for checking the quality of TS-related calculations, contains helper functions for Scheduler.
"""

import logging
import os

import numpy as np
from typing import TYPE_CHECKING, List, Optional

from arc import parser
from arc.common import (ARC_PATH,
                        convert_list_index_0_to_1,
                        get_logger,
                        read_yaml_file,
                        )

if TYPE_CHECKING:
    from arc.job.adapter import JobAdapter
    from arc.species.species import ARCSpecies, TSGuess
    from arc.reaction import ARCReaction

logger = get_logger()


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
    reaction.check_ts(verbose=False, rxn_zone_atom_indices=rxn_zone_atom_indices)
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
