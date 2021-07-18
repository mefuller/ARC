"""
A module for atom-mapping a species or a set of species.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from qcelemental.exceptions import ValidationError
from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.data.kinetics.family import TemplateReaction
from rmgpy.exceptions import ForbiddenStructureException
from rmgpy.molecule import Molecule
from rmgpy.reaction import same_species_lists
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import logger
from arc.species import ARCSpecies
from arc.species.converter import translate_xyz, xyz_to_str


if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import KineticsFamily
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction


def map_h_abstraction(rxn: 'ARCReaction') -> Optional[List[int]]:
    """
    Map a hydrogen abstraction reaction.
    This function does not populate the rxn.family attribute, but does checks and expects it to be populated.

    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.

    Returns:
        Optional[List[int]]:
            Entry values are atom indices in the products mapped to the respective reactant atom (entry index).
    """
    if rxn.family is None:
        rmgdb.determine_family(rxn)
    if rxn.family is None:
        return None
    if rxn.family.label != 'H_Abstraction':
        raise ValueError(f'Only H_Abstraction reactions are supported by this function, got a {rxn.family} reaction.')

    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction=rxn)
    r_dict, p_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                          rmg_reaction=rmg_reactions[0])

    r_h_index = r_dict['*2']
    p_h_index = p_dict['*2']
    len_r1, len_p1 = rxn.r_species[0].number_of_atoms, rxn.p_species[0].number_of_atoms
    r1_h2 = 0 if r_h_index < len_r1 else 1  # Identify R(*1)-H(*2), it's either reactant 0 or reactant 1.
    r3 = 0 if r1_h2 else 1  # Identify R(*3) in the reactants.
    r3_h2 = 0 if p_h_index < len_p1 else 1  # Identify R(*3)-H(*2), it's either product 0 or product 1.
    r1 = 0 if r3_h2 else 1  # Identify R(*1) in the products.

    spc_r1_h2 = ARCSpecies(label='R1-H2',
                           mol=rxn.r_species[r1_h2].mol,
                           xyz=rxn.r_species[r1_h2].get_xyz(),
                           bdes=[(r_dict['*1'] - r1_h2 * len_r1 + 1,
                                  r_dict['*2'] - r1_h2 * len_r1 + 1)],  # Mark the R(*1)-H(*2) bond for scission.
                           )
    spc_r1_h2.final_xyz = spc_r1_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r1_h2_cut = [spc for spc in spc_r1_h2.scissors() if spc.label != 'H'][0]
    spc_r3_h2 = ARCSpecies(label='R3-H2',
                           mol=rxn.p_species[r3_h2].mol,
                           xyz=rxn.p_species[r3_h2].get_xyz(),
                           bdes=[(p_dict['*3'] - r3_h2 * len_p1 + 1,
                                  p_dict['*2'] - r3_h2 * len_p1 + 1)],  # Mark the R(*3)-H(*2) bond for scission.
                           )
    spc_r3_h2.final_xyz = spc_r3_h2.get_xyz()  # Scissors require the .final_xyz attribute to be populated.
    spc_r3_h2_cut = [spc for spc in spc_r3_h2.scissors() if spc.label != 'H'][0]
    map_1 = map_two_species(spc_r1_h2_cut, rxn.p_species[r1])
    map_2 = map_two_species(rxn.r_species[r3], spc_r3_h2_cut)

    result = {r_h_index: p_h_index}
    r_increment = r1_h2 * len_r1
    p_increment = (1 - r3_h2) * len_p1
    for i, entry in enumerate(map_1):
        r_index = i + r_increment + int(i + r_increment >= r_h_index)
        p_index = entry + p_increment
        result[r_index] = p_index
    r_increment = (1 - r1_h2) * len_r1
    p_increment = r3_h2 * len_p1
    for i, entry in enumerate(map_2):
        r_index = i + r_increment
        p_index = entry + p_increment + int(i + p_increment >= p_h_index)
        result[r_index] = p_index
    return [val for key, val in sorted(result.items(), key=lambda item: item[0])]


def map_two_species(spc_1: Union[ARCSpecies, Species, Molecule],
                    spc_2: Union[ARCSpecies, Species, Molecule],
                    map_type: str = 'list',
                    ) -> Optional[Union[List[int], Dict[int, int]]]:
    """
    Map the atoms in spc1 to the atoms in spc2.
    All indices are 0-indexed.
    If a dict type atom map is returned, it cold conveniently be used to map ``spc_2`` -> ``spc_1`` by doing::

        ordered_spc1.atoms = [spc_2.atoms[atom_map[i]] for i in range(len(spc_2.atoms))]

    Args:
        spc_1 (Union[ARCSpecies, Species, Molecule]): Species 1.
        spc_2 (Union[ARCSpecies, Species, Molecule]): Species 2.
        map_type (str, optional): Whether to return a 'list' or a 'dict' map type.

    Returns:
        Optional[Union[List[int], Dict[int, int]]]:
            The atom map. By default, a list is returned.
            If the map is of list type, entry indices are atom indices of ``spc_1``, entry values are atom indices of ``spc_2``.
            If the map is of dict type, keys are atom indices of ``spc_1``, values are atom indices of ``spc_2``.
    """
    qcmol_1 = create_qc_mol(species=spc_1.copy())
    qcmol_2 = create_qc_mol(species=spc_2.copy())
    if qcmol_1 is None or qcmol_2 is None:
        return None
    if len(qcmol_1.symbols) != len(qcmol_2.symbols):
        raise ValueError(f'The number of atoms in spc1 ({spc_1.number_of_atoms}) must be equal '
                         f'to the number of atoms in spc1 ({spc_2.number_of_atoms}).')
    data = qcmol_2.align(ref_mol=qcmol_1, verbose=0)[1]  # not problematic
    atom_map = data['mill'].atommap.tolist()  # not problematic
    if map_type == 'dict':  # ** Todo: test
        atom_map = {key: val for key, val in enumerate(atom_map)}
    return atom_map


def create_qc_mol(species: Union[ARCSpecies, Species, Molecule, List[Union[ARCSpecies, Species, Molecule]]],
                  charge: Optional[int] = None,
                  multiplicity: Optional[int] = None,
                  ) -> Optional[QCMolecule]:
    """
    Create a single QCMolecule object instance from a ARCSpecies object instances.

    Args:
        species (List[Union[ARCSpecies, Species, Molecule]]): Entries are ARCSpecies / RMG Species / RMG Molecule
                                                              object instances.
        charge (int, optional): The overall charge of the surface.
        multiplicity (int, optional): The overall electron multiplicity of the surface.

    Returns:
        Optional[QCMolecule]: The respective QCMolecule object instance.
    """
    species = [species] if not isinstance(species, list) else species
    species_list = list()
    for spc in species:
        if isinstance(spc, ARCSpecies):
            species_list.append(spc)
        elif isinstance(spc, Species):
            species_list.append(ARCSpecies(label='S', mol=spc.molecule[0]))
        elif isinstance(spc, Molecule):
            species_list.append(ARCSpecies(label='S', mol=spc))
        else:
            raise ValueError(f'Species entries may only be ARCSpecies, RMG Species, or RMG Molecule, '
                             f'got {spc} which is a {type(spc)}.')
    if len(species_list) == 1:
        if charge is None:
            charge = species_list[0].charge
        if multiplicity is None:
            multiplicity = species_list[0].multiplicity
    if charge is None or multiplicity is None:
        raise ValueError(f'An overall charge and multiplicity must be specified for multiple species, '
                         f'got: {charge} and {multiplicity}, respectively')
    radius = max(spc.radius for spc in species_list)
    qcmol = None
    try:
        qcmol = QCMolecule.from_data(
            data='\n--\n'.join([xyz_to_str(translate_xyz(spc.get_xyz(), translation=(i * radius, 0, 0)))
                                for i, spc in enumerate(species_list)]),
            molecular_charge=charge,
            molecular_multiplicity=multiplicity,
            fragment_charges=[spc.charge for spc in species_list],
            fragment_multiplicities=[spc.multiplicity for spc in species_list],
            orient=False,
        )
    except ValidationError as err:
        logger.warning(f'Could not get atom map for {[spc.label for spc in species_list]}, got:\n{err}')
    return qcmol


def get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction: 'ARCReaction',
                                                         rmg_reaction: TemplateReaction,
                                                         ) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
    """
    Get the RMG reaction labels and the corresponding 0-indexed atom indices
    for all labeled atoms in a TemplateReaction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (TemplateReaction): A respective RMG family TemplateReaction object instance.

    Returns:
        Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]]]:
            The tuple entries relate to reactants and products.
            Keys are labels (e.g., '*1'), values are corresponding 0-indices atoms.
    """
    if not hasattr(rmg_reaction, 'labeledAtoms') or not rmg_reaction.labeledAtoms:
        return None, None

    for mol in rmg_reaction.reactants + rmg_reaction.products:
        mol.generate_resonance_structures(save_order=True)

    r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction)

    reactant_index_dict, product_index_dict = dict(), dict()
    reactant_atoms, product_atoms = list(), list()
    rmg_reactant_order = [val[0] for key, val in sorted(r_map.items(), key=lambda item: item[0])]
    rmg_product_order = [val[0] for key, val in sorted(p_map.items(), key=lambda item: item[0])]
    for i in rmg_reactant_order:
        reactant_atoms.extend([atom for atom in rmg_reaction.reactants[i].atoms])
    for i in rmg_product_order:
        product_atoms.extend([atom for atom in rmg_reaction.products[i].atoms])
    for reactant in rmg_reaction.reactants:
        print('r labeled atoms:')
        print(rmg_reaction.labeledAtoms)
        for label, atom in rmg_reaction.labeledAtoms['reactants']:
            for i, reactant_atom in enumerate(reactant_atoms):
                if reactant_atom.id == atom.id:
                    reactant_index_dict[label] = i
    for product in rmg_reaction.products:
        print('p labeled atoms:')
        print(product.get_all_labeled_atoms())
        for label, atom in rmg_reaction.labeledAtoms['products']:
            for i, product_atom in enumerate(product_atoms):
                if product_atom.id == atom.id:
                    product_index_dict[label] = i


    # for labeled_atom in rmg_reaction.labeledAtoms:
    #     print(labeled_atom)
    #     for atom_list, index_dict in zip([reactant_atoms, product_atoms], [reactant_index_dict, product_index_dict]):
    #         for i, atom in enumerate(atom_list):
    #             if atom.id == labeled_atom[1].id:
    #                 print(atom.id, labeled_atom[0])
    #                 index_dict[labeled_atom[0]] = i
    #                 break
    return reactant_index_dict, product_index_dict


def map_arc_rmg_species(arc_reaction: 'ARCReaction',
                        rmg_reaction: Union['Reaction', TemplateReaction],
                        concatenate: bool = True,
                        ) -> Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
    """
    Map the species pairs in an ARC reaction to those in a respective RMG reaction
    which is defined in the same direction.

    Args:
        arc_reaction (ARCReaction): An ARCReaction object instance.
        rmg_reaction (Union[Reaction, TemplateReaction]): A respective RMG family TemplateReaction object instance.
        concatenate (bool, optional): Whether to return isomorphic species as a single list (``True``),
                                      or to return isomorphic species separately.

    Returns:
        Tuple[Dict[int, Union[List[int], int]], Dict[int, Union[List[int], int]]]:
            The first tuple entry refers to reactants, the second to products.
            Keys are specie indices in the ARC reaction,
            values are respective indices in the RMG reaction.
            If ``concatenate`` is ``True``, values are lists of integers. Otherwise, values are integers.
    """
    if rmg_reaction.is_isomerization():
        if concatenate:
            return {0: [0]}, {0: [0]}
        else:
            return {0: 0}, {0: 0}
    r_map, p_map = dict(), dict()
    arc_reactants, arc_products = arc_reaction.get_reactants_and_products(arc=True)
    for spc_map, rmg_species, arc_species in [(r_map, rmg_reaction.reactants, arc_reactants),
                                              (p_map, rmg_reaction.products, arc_products)]:
        for i, arc_spc in enumerate(arc_species):
            for j, rmg_obj in enumerate(rmg_species):
                if isinstance(rmg_obj, Molecule):
                    rmg_spc = Species(molecule=[rmg_obj])
                elif isinstance(rmg_obj, Species):
                    rmg_spc = rmg_obj
                else:
                    raise ValueError(f'Expected an RMG object instance of Molecule() or Species(),'
                                     f'got {rmg_obj} which is a {type(rmg_obj)}.')
                rmg_spc.generate_resonance_structures(save_order=True)
                if rmg_spc.is_isomorphic(arc_spc.mol, save_order=True):
                    if i in spc_map.keys() and concatenate:  # ** Todo: test
                        spc_map[i].append(j)
                    elif concatenate:
                        spc_map[i] = [j]
                    else:
                        spc_map[i] = j
    return r_map, p_map


def find_equivalent_atoms_in_reactants(arc_reaction: 'ARCReaction') -> Optional[List[List[int]]]:
    """
    Find atom indices that are equivalent in the reactants of an ARCReaction
    in the sense that they represent degenerate reaction sites that are indifferentiable in 2D.
    Bridges between RMG reaction templates and ARC's 3D TS structures.
    Running indices in the returned structure relate to reactant_0 + reactant_1 + ...

    Args:
        arc_reaction ('ARCReaction'): The ARCReaction object instance.

    Returns:
        Optional[List[List[int]]]: Entries are lists of 0-indices, each such list represents equivalent atoms.
    """
    rmg_reactions = _get_rmg_reactions_from_arc_reaction(arc_reaction)
    dicts = [get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reaction=rmg_reaction,
                                                                  arc_reaction=arc_reaction)[0]
             for rmg_reaction in rmg_reactions]
    equivalence_map = dict()
    for index_dict in dicts:
        for key, value in index_dict.items():
            if key in equivalence_map:
                equivalence_map[key].append(value)
            else:
                equivalence_map[key] = [value]
    equivalent_indices = list(list(set(equivalent_list)) for equivalent_list in equivalence_map.values())
    return equivalent_indices


def _get_rmg_reactions_from_arc_reaction(arc_reaction: 'ARCReaction') -> Optional[List[TemplateReaction]]:
    """
    A helper function for getting RMG reactions from an ARC reaction.

    Args:
        arc_reaction (ARCReaction): The ARCReaction object instance.

    Returns:
        Optional[List[TemplateReaction]]:
            The respective RMG TemplateReaction object instances (considering resonance structures).
    """
    if arc_reaction.family is None:
        rmgdb.determine_family(arc_reaction)
    if arc_reaction.family is None:
        return None
    rmg_reactions = generate_reactions(family=arc_reaction.family,
                                       reactants=[spc.mol for spc in arc_reaction.r_species],
                                       products=[spc.mol for spc in arc_reaction.p_species],
                                       prod_resonance=True,
                                       )
    for rmg_reaction in rmg_reactions:
        for reactant in rmg_reaction.reactants:
            print([atom.label for atom in reactant.atoms])
        for product in rmg_reaction.products:
            print([atom.label for atom in product.atoms])
        r_map, p_map = map_arc_rmg_species(arc_reaction=arc_reaction, rmg_reaction=rmg_reaction, concatenate=False)
        print(r_map, p_map)
        ordered_rmg_reactants = [rmg_reaction.reactants[r_map[i]] for i in range(len(rmg_reaction.reactants))]
        ordered_rmg_products = [rmg_reaction.products[p_map[i]] for i in range(len(rmg_reaction.products))]
        mapped_rmg_reactants, mapped_rmg_products = list(), list()
        for ordered_rmg_mols, arc_species, mapped_mols in zip([ordered_rmg_reactants, ordered_rmg_products],
                                                              [arc_reaction.r_species, arc_reaction.p_species],
                                                              [mapped_rmg_reactants, mapped_rmg_products],
                                                              ):
            for rmg_mol, arc_spc in zip(ordered_rmg_mols, arc_species):
                mol = arc_spc.copy().mol
                atom_map = map_two_species(mol, rmg_mol, map_type='dict')
                new_atoms_list = list()
                for i in range(len(rmg_mol.atoms)):
                    rmg_mol.atoms[atom_map[i]].id = mol.atoms[i].id
                    new_atoms_list.append(rmg_mol.atoms[atom_map[i]])
                rmg_mol.atoms = new_atoms_list
                mapped_mols.append(rmg_mol)
        rmg_reaction.reactants, rmg_reaction.products = mapped_rmg_reactants, mapped_rmg_products
    return rmg_reactions


def generate_reactions(family: 'KineticsFamily',
                       reactants: Union[List[Molecule], List[List[Molecule]]],
                       products: Optional[Union[List[Molecule], List[List[Molecule]]]] = None,
                       prod_resonance: bool = True,
                       ) -> List[TemplateReaction]:
    """
    Generate all reactions between the provided list of one, two, or three
    ``reactants``, which should be either single :class:`Molecule` objects
    or lists of same. Does not estimate the kinetics of these reactions
    at this time. Returns a list of :class:TemplateReaction objects
    using :class:`Molecule` objects for both reactants and products
    The reactions are constructed such that the forward direction is
    consistent with the template of this reaction family.

    This function was inspired by RMG's KineticsFamily.generate_reactions().

    Args:
        family (KineticsFamily): THe KineticsFamily object instance corresponding to the desired reaction family.
        reactants (Union[List[Molecule], List[List[Molecule]]]): Entries are reactant Molecule object instances.
        products (Union[List[Molecule], List[List[Molecule]]], optional): Entries are product Molecule object instances.
        prod_resonance (bool, optional): Whether to generate resonance structures for product checking.

    Returns:
        List[TemplateReaction]: Entries are all the reactions containing the ``Molecule`` object instances with the
                                specified reactants (and products, if requested) within the relevant ``family``.
                                Degenerate reactions are returned as separate reactions.
    """
    reaction_list = list()
    reactants = [reactant if isinstance(reactant, list) else [reactant] for reactant in reactants]
    reaction_list.extend(_generate_reactions(family=family,
                                             reactants=reactants,
                                             products=products,
                                             forward=True,
                                             prod_resonance=prod_resonance,
                                             ))
    if not family.own_reverse and family.reversible:
        reaction_list.extend(_generate_reactions(family=family,
                                                 reactants=reactants,
                                                 products=products,
                                                 forward=False,
                                                 prod_resonance=prod_resonance,
                                                 ))
    return reaction_list


def _generate_reactions(family: 'KineticsFamily',
                        reactants: List[List[Molecule]],
                        products: Optional[List[List[Molecule]]] = None,
                        forward: bool = True,
                        prod_resonance: bool = True,
                        ) -> List[TemplateReaction]:
    """
    Generate a list of all the possible reactions of a certain ``family`` between
    the list of ``reactants`` that yield the ``products``. The number of reactants
    provided must match the number of reactants expected by the template, or an
    empty list is returned. Each item in the list of reactants should
    be a list of :class:`Molecule` objects, each representing a resonance
    structure of the species of interest.

    This function returns all reactions, and degenerate reactions can then be
    found using ``rmgpy.data.kinetics.common.find_degenerate_reactions``.

    This function was inspired by RMG's KineticsFamily._generate_reactions().

    Args:
        family (KineticsFamily): THe KineticsFamily object instance corresponding to the desired reaction family.
        reactants (List[List[Molecule]]): Entries are lists of reactant Molecule object instances.
        products (List[List[Molecule]], optional): Entries are lists of product Molecule object instances.
        forward (bool, optional): Whether the forward or reverse reaction template should be applied.
        prod_resonance (bool, optional): Whether to generate resonance structures for product checking.

    Returns:
        List[TemplateReaction]: Entries are all the reactions containing the ``Molecule`` object instances with the
                                specified reactants (and products, if requested) within the relevant ``family``.
                                Degenerate reactions are returned as separate reactions.
    """
    rxn_list = list()

    if not forward and family.reverse_template is None:
        return list()
    template = family.forward_template if forward else family.reverse_template
    reactant_num = family.reactant_num if forward else family.product_num

    if family.auto_generated and reactant_num != len(reactants):
        return list()

    if len(reactants) > len(template.reactants):
        # if the family has one template and is bimolecular, split template into multiple reactants
        try:
            groups = template.reactants[0].item.split()
            template_reactants = list()
            for group in groups:
                template_reactants.append(group)
        except AttributeError:
            template_reactants = [x.item for x in template.reactants]
    else:
        template_reactants = [x.item for x in template.reactants]

    # Unimolecular reactants: A --> products
    if len(reactants) == 1 and len(template_reactants) == 1:
        for molecule in reactants[0]:
            mappings = family._match_reactant_to_template(molecule, template_reactants[0])
            for mapping in mappings:
                reactant_structures = [molecule]
                try:
                    product_structures = family._generate_product_structures(reactant_structures=reactant_structures,
                                                                             maps=[mapping],
                                                                             forward=forward,
                                                                             relabel_atoms=False,
                                                                             )
                except ForbiddenStructureException:
                    pass
                else:
                    if product_structures is not None:
                        rxn = create_reaction(family=family,
                                              reactants=reactant_structures,
                                              products=product_structures,
                                              forward=forward,
                                              )
                        if rxn is not None:
                            rxn_list.append(rxn)

    # Bimolecular reactants: A + B --> products
    elif len(reactants) == 2 and len(template_reactants) == 2:
        molecules_a, molecules_b = reactants[0], reactants[1]
        for molecule_a in molecules_a:
            for molecule_b in molecules_b:
                # Reactants stored as A + B
                mappings_a = family._match_reactant_to_template(molecule_a, template_reactants[0])
                mappings_b = family._match_reactant_to_template(molecule_b, template_reactants[1])

                for map_a in mappings_a:
                    for map_b in mappings_b:
                        # Reverse the order of reactants in case we have a family with only one reactant tree
                        # that can produce different products depending on the order of reactants.
                        reactant_structures = [molecule_b, molecule_a]
                        try:
                            product_structures = family._generate_product_structures(reactant_structures=reactant_structures,
                                                                                     maps=[map_b, map_a],
                                                                                     forward=forward,
                                                                                     relabel_atoms=False,
                                                                                     )
                        except ForbiddenStructureException:
                            pass
                        else:
                            if product_structures is not None:
                                rxn = create_reaction(family=family,
                                                      reactants=reactant_structures,
                                                      products=product_structures,
                                                      forward=forward,
                                                      )
                                if rxn is not None:
                                    rxn_list.append(rxn)

                # Only check for swapped reactants if they are different.
                if reactants[0] is not reactants[1]:
                    # Reactants stored as B + A.
                    mappings_a = family._match_reactant_to_template(molecule_a, template_reactants[1])
                    mappings_b = family._match_reactant_to_template(molecule_b, template_reactants[0])
                    # Iterate over each pair of matches (A, B).
                    for map_a in mappings_a:
                        for map_b in mappings_b:
                            reactant_structures = [molecule_a, molecule_b]
                            try:
                                product_structures = family._generate_product_structures(reactant_structures=reactant_structures,
                                                                                         maps=[map_a, map_b],
                                                                                         forward=forward,
                                                                                         relabel_atoms=False,
                                                                                         )
                            except ForbiddenStructureException:
                                pass
                            else:
                                if product_structures is not None:
                                    rxn = create_reaction(family=family,
                                                          reactants=reactant_structures,
                                                          products=product_structures,
                                                          forward=forward,
                                                          )
                                    if rxn is not None:
                                        rxn_list.append(rxn)

    elif len(reactants) == 3 and len(template_reactants) == 3:
            molecules_a, molecules_b, molecules_c = reactants[0], reactants[1], reactants[2]
            for molecule_a in molecules_a:
                for molecule_b in molecules_b:
                    for molecule_c in molecules_c:
                        def generate_products_and_reactions(order):
                            """
                            A helper function to generate products and reactions.
                            If ``order`` is (0, 1, 2), it corresponds to reactants stored as A + B + C, etc.
                            """
                            _mappings_a = family._match_reactant_to_template(molecule_a, template_reactants[order[0]])
                            _mappings_b = family._match_reactant_to_template(molecule_b, template_reactants[order[1]])
                            _mappings_c = family._match_reactant_to_template(molecule_c, template_reactants[order[2]])

                            # Iterate over each pair of matches (A, B, C)
                            for _map_a in _mappings_a:
                                for _map_b in _mappings_b:
                                    for _map_c in _mappings_c:
                                        _reactant_structures = [molecule_a, molecule_b, molecule_c]
                                        _maps = [_map_a, _map_b, _map_c]
                                        # Reorder reactants in case we have a family with fewer reactant trees than
                                        # reactants and different reactant orders can produce different products
                                        _reactant_structures = [_reactant_structures[_i] for _i in order]
                                        _maps = [_maps[_i] for _i in order]
                                        try:
                                            _product_structures = family._generate_product_structures(
                                                reactant_structures=_reactant_structures,
                                                maps=_maps,
                                                forward=forward,
                                                relabel_atoms=False,
                                            )
                                        except ForbiddenStructureException:
                                            pass
                                        else:
                                            if _product_structures is not None:
                                                _rxn = create_reaction(family=family,
                                                                       reactants=_reactant_structures,
                                                                       products=_product_structures,
                                                                       forward=forward,
                                                                       )
                                                if _rxn is not None:
                                                    rxn_list.append(_rxn)

                        # Reactants stored as A + B + C
                        generate_products_and_reactions((0, 1, 2))
                        # Only check for swapped reactants if they are different
                        if reactants[1] is not reactants[2]:
                            # Reactants stored as A + C + B
                            generate_products_and_reactions((0, 2, 1))
                        if reactants[0] is not reactants[1]:
                            # Reactants stored as B + A + C
                            generate_products_and_reactions((1, 0, 2))
                        if reactants[0] is not reactants[2]:
                            # Reactants stored as C + B + A
                            generate_products_and_reactions((2, 1, 0))
                            if reactants[0] is not reactants[1] and reactants[1] is not reactants[2]:
                                # Reactants stored as C + A + B
                                generate_products_and_reactions((2, 0, 1))
                                # Reactants stored as B + C + A
                                generate_products_and_reactions((1, 2, 0))

    # If ``products`` is given, remove reactions from the reaction list that don't generate the requested products.
    if products is not None:
        rxn_list_0 = rxn_list[:]
        rxn_list = list()
        for reaction in rxn_list_0:
            products_0 = reaction.products if forward else reaction.reactants
            # Only keep reactions which give the requested products
            # If prod_resonance=True, then use strict=False to consider all resonance structures
            if same_species_lists(products, products_0, strict=not prod_resonance, save_order=True):
                rxn_list.append(reaction)

    for reaction in rxn_list:
        # # Restore the labeled atoms long enough to generate some metadata
        # for reactant in reaction.reactants:
        #     reactant.clear_labeled_atoms()
        # for label, atom in reaction.labeledAtoms:
        #     if isinstance(atom, list):
        #         for atm in atom:
        #             atm.label = label
        #     else:
        #         atom.label = label
        reaction.reversible = family.reversible

    return rxn_list


def create_reaction(family: 'KineticsFamily',
                    reactants: List[Molecule],
                    products: List[Molecule] = None,
                    forward: bool = True,
                    ) -> Optional[TemplateReaction]:
    """
    Create and return a new ``TemplateReaction`` object instance containing the
    provided ``reactants`` and ``products``.

    This function was inspired by RMG's KineticsFamily._create_reaction().
    """
    if same_species_lists(reactants, products, save_order=True):
        return None
    reaction = TemplateReaction(
        reactants=reactants if forward else products,
        products=products if forward else reactants,
        degeneracy=1,
        reversible=family.reversible,
        family=family.label,
        is_forward=forward,
    )
    reaction.labeledAtoms = {'reactants': list(), 'products': list()}
    for reactant in reaction.reactants:
        for label, atom in reactant.get_all_labeled_atoms().items():
            reaction.labeledAtoms['reactants'].append((label, atom))
    for product in reaction.products:
        for label, atom in product.get_all_labeled_atoms().items():
            reaction.labeledAtoms['products'].append((label, atom))
    return reaction
