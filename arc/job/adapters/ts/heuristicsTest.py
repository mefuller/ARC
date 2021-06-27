#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.heuristics module
"""

import os
import unittest
import shutil

from rmgpy.molecule.molecule import Molecule
from rmgpy.reaction import Reaction
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import ARC_PATH
from arc.job.adapters.ts.heuristics import HeuristicsAdapter, find_distant_neighbor, reverse_xyz
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies


class TestHeuristicsAdapter(unittest.TestCase):
    """
    Contains unit tests for the HeuristicsAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)

    def test_heuristics_for_h_abstraction(self):
        """
        Test that ARC can generate TS guesses based on heuristics for H Abstraction reactions.
        """
        ch4_xyz = """C 0.0000000 0.0000000 0.0000000
H 0.6279670 0.6279670 0.6279670
H -0.6279670 -0.6279670 0.6279670
H -0.6279670 0.6279670 -0.6279670
H 0.6279670 -0.6279670 -0.6279670"""
        h_xyz = """H 0.0 0.0 0.0"""
        ch3_xyz = """C 0.0000000 0.0000000 0.0000000
H 0.0000000 1.0922900 0.0000000
H 0.9459510 -0.5461450 0.0000000
H -0.9459510 -0.5461450 0.0000000"""
        h2_xyz = """H 0.0000000 0.0000000 0.3714780
H 0.0000000 0.0000000 -0.3714780"""
        ch4 = ARCSpecies(label='CH4', smiles='C', xyz=ch4_xyz)
        h = ARCSpecies(label='H', smiles='[H]', xyz=h_xyz)
        ch3 = ARCSpecies(label='CH3', smiles='[CH3]', xyz=ch3_xyz)
        h2 = ARCSpecies(label='H2', smiles='[H][H]', xyz=h2_xyz)
        rxn1 = ARCReaction(reactants=['CH4', 'H'], products=['CH3', 'H2'],
                           r_species=[ch4, h], p_species=[ch3, h2],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('C'),
                                                            Species().from_smiles('[H]')],
                                                 products=[Species().from_smiles('[CH3]'),
                                                           Species().from_smiles('[H][H]')]))
        rxn1.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn1.family.label, 'H_Abstraction')
        heuristics1 = HeuristicsAdapter(job_type='tsg',
                                        reactions=[rxn1],
                                        testing=True,
                                        project='test',
                                        project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                        )
        heuristics1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 1)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[0].initial_xyz['coords']), 6)
        self.assertTrue(rxn1.ts_species.ts_guesses[0].success)

        c3h8_xyz = """C	0.0000000 0.0000000 0.5949240
C 0.0000000 1.2772010 -0.2630030
C 0.0000000 -1.2772010 -0.2630030
H 0.8870000 0.0000000 1.2568980
H -0.8870000 0.0000000 1.2568980
H 0.0000000 2.1863910 0.3643870
H 0.0000000 -2.1863910 0.3643870
H 0.8933090 1.3136260 -0.9140200
H -0.8933090 1.3136260 -0.9140200
H -0.8933090 -1.3136260 -0.9140200
H 0.8933090 -1.3136260 -0.9140200"""
        ho2_xyz = """O 0.0553530 -0.6124600 0.0000000
O 0.0553530 0.7190720 0.0000000
H -0.8856540 -0.8528960 0.0000000"""
        c3h7_xyz = """C 1.3077700 -0.2977690 0.0298660
C 0.0770610 0.5654390 -0.0483740
C -1.2288150 -0.2480100 0.0351080
H -2.1137100 0.4097560 -0.0247200
H -1.2879330 -0.9774520 -0.7931500
H -1.2803210 -0.8079420 0.9859990
H 0.1031750 1.3227340 0.7594170
H 0.0813910 1.1445730 -0.9987260
H 2.2848940 0.1325040 0.2723890
H 1.2764100 -1.3421290 -0.3008110"""
        h2o2_xyz = """O 0.0000000 0.7275150 -0.0586880
O 0.0000000 -0.7275150 -0.0586880
H 0.7886440 0.8942950 0.4695060
H -0.7886440 -0.8942950 0.4695060"""
        c3h8 = ARCSpecies(label='C3H8', smiles='CCC', xyz=c3h8_xyz)
        ho2 = ARCSpecies(label='HO2', smiles='O[O]', xyz=ho2_xyz)
        c3h7 = ARCSpecies(label='C3H7', smiles='[CH2]CC', xyz=c3h7_xyz)
        h2o2 = ARCSpecies(label='H2O2', smiles='OO', xyz=h2o2_xyz)
        rxn2 = ARCReaction(reactants=['C3H8', 'HO2'], products=['C3H7', 'H2O2'],
                           r_species=[c3h8, ho2], p_species=[c3h7, h2o2],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('CCC'),
                                                            Species().from_smiles('O[O]')],
                                                 products=[Species().from_smiles('[CH2]CC'),
                                                           Species().from_smiles('OO')]))
        rxn2.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn2.family.label, 'H_Abstraction')
        heuristics2 = HeuristicsAdapter(job_type='tsg',
                                        reactions=[rxn2],
                                        testing=True,
                                        project='test',
                                        project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                        )
        heuristics2.execute_incore()
        self.assertTrue(rxn2.ts_species.is_ts)
        self.assertEqual(rxn2.ts_species.charge, 0)
        self.assertEqual(rxn2.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn2.ts_species.ts_guesses), 18)
        self.assertEqual(rxn2.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O'))
        self.assertEqual(rxn2.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O', 'O'))
        self.assertEqual(len(rxn2.ts_species.ts_guesses[1].initial_xyz['coords']), 14)
        self.assertTrue(rxn2.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[1].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[2].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[3].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[4].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[5].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[6].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[7].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[8].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[9].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[10].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[11].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[12].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[13].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[14].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[15].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[16].success)
        self.assertTrue(rxn2.ts_species.ts_guesses[17].success)

        cccoh_xyz = """C -1.4562640 1.2257490 0.0000000
C 0.0000000 0.7433860 0.0000000
C 0.1008890 -0.7771710 0.0000000
O 1.4826600 -1.1256940 0.0000000
H -1.5081640 2.3212940 0.0000000
H -1.9909330 0.8624630 0.8882620
H -1.9909330 0.8624630 -0.8882620
H 0.5289290 1.1236530 0.8845120
H 0.5289290 1.1236530 -0.8845120
H -0.4109400 -1.1777970 0.8923550
H -0.4109400 -1.1777970 -0.8923550
H 1.5250230 -2.0841670 0.0000000"""
        oh_xyz = """O 0.0000000 0.0000000 0.1078170
H 0.0000000 0.0000000 -0.8625320"""
        ccco_xyz = """C      -1.22579665    0.34157501   -0.08330600
C      -0.04626439   -0.57243496    0.22897599
C      -0.11084721   -1.88672335   -0.59040103
O       0.94874959   -2.60335587   -0.24842497
H      -2.17537216   -0.14662734    0.15781317
H      -1.15774972    1.26116047    0.50644174
H      -1.23871523    0.61790236   -1.14238547
H       0.88193016   -0.02561912    0.01201028
H      -0.05081615   -0.78696747    1.30674288
H      -1.10865982   -2.31155703   -0.39617740
H      -0.21011639   -1.57815495   -1.64338139"""
        h2o_xyz = """O      -0.00032832    0.39781490    0.00000000
H      -0.76330345   -0.19953755    0.00000000
H       0.76363177   -0.19827735    0.00000000"""
        cccoh = ARCSpecies(label='CCCOH', smiles='CCCO', xyz=cccoh_xyz)
        oh = ARCSpecies(label='OH', smiles='[OH]', xyz=oh_xyz)
        ccco = ARCSpecies(label='CCCO', smiles='CCC[O]', xyz=ccco_xyz)
        h2o = ARCSpecies(label='H2O', smiles='O', xyz=h2o_xyz)
        rxn3 = ARCReaction(reactants=['CCCOH', 'OH'], products=['CCCO', 'H2O'],
                           r_species=[cccoh, oh], p_species=[ccco, h2o],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('CCCO'),
                                                            Species().from_smiles('[OH]')],
                                                 products=[Species().from_smiles('CCC[O]'),
                                                           Species().from_smiles('O')]))
        rxn3.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn3.family.label, 'H_Abstraction')
        heuristics3 = HeuristicsAdapter(job_type='tsg',
                                        reactions=[rxn3],
                                        testing=True,
                                        project='test',
                                        project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                        )
        heuristics3.execute_incore()
        self.assertTrue(rxn3.ts_species.is_ts)
        self.assertEqual(rxn3.ts_species.charge, 0)
        self.assertEqual(rxn3.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn3.ts_species.ts_guesses), 18)
        self.assertEqual(rxn3.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'O'))
        self.assertEqual(len(rxn3.ts_species.ts_guesses[1].initial_xyz['coords']), 14)

        cdcoh_xyz = """C      -0.80601307   -0.11773769    0.32792128
C       0.23096883    0.47536513   -0.26437348
O       1.44620485   -0.11266560   -0.46339257
H      -1.74308628    0.41660480    0.45016601
H      -0.75733964   -1.13345488    0.70278513
H       0.21145717    1.48838416   -0.64841675
H       1.41780836   -1.01649567   -0.10468897"""
        cdco_xyz = """C      -0.68324480   -0.04685539   -0.10883672
C       0.63642204    0.05717653    0.10011041
O       1.50082619   -0.82476680    0.32598015
H      -1.27691852    0.84199331   -0.29048852
H      -1.17606821   -1.00974165   -0.10030145
H       0.99232452    1.08896899    0.06242974"""
        cdcoh = ARCSpecies(label='C=COH', smiles='C=CO', xyz=cdcoh_xyz)
        cdco = ARCSpecies(label='C=CO', smiles='C=C[O]', xyz=cdco_xyz)
        rxn4 = ARCReaction(reactants=['C=COH', 'H'], products=['C=CO', 'H2'],
                           r_species=[cdcoh, h], p_species=[cdco, h2],
                           rmg_reaction=Reaction(reactants=[Species().from_smiles('C=CO'),
                                                            Species().from_smiles('[H]')],
                                                 products=[Species().from_smiles('C=C[O]'),
                                                           Species().from_smiles('[H][H]')]))
        rxn4.determine_family(rmg_database=self.rmgdb)
        self.assertEqual(rxn4.family.label, 'H_Abstraction')
        heuristics4 = HeuristicsAdapter(job_type='tsg',
                                        reactions=[rxn4],
                                        testing=True,
                                        project='test',
                                        project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'),
                                        )
        heuristics4.execute_incore()
        self.assertTrue(rxn4.ts_species.is_ts)
        self.assertEqual(rxn4.ts_species.charge, 0)
        self.assertEqual(rxn4.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn4.ts_species.ts_guesses), 1)
        self.assertEqual(rxn4.ts_species.ts_guesses[0].initial_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))

    def test_reverse_xyz(self):
        """Test the reverse_xyz() function."""
        xyz = {'symbols': ('S', 'O', 'N', 'C', 'H', 'H', 'H'),
               'isotopes': (32, 16, 14, 12, 1, 1, 1),
               'coords': ((1.02558264, -0.04344404, -0.07343859),
                          (-0.25448248, 1.10710477, 0.18359696),
                          (-1.30762173, 0.15796567, -0.1048929),
                          (-0.49011438, -1.0370438, 0.15365747),
                          (-0.6486995, -1.85796321, -0.54773423),
                          (-0.60359153, -1.37304859, 1.18613964),
                          (-1.43009127, 0.23517346, -1.11797908))}

        reversed_xyz_1 = reverse_xyz(xyz=xyz, reactants_reversed=False, rmg_reactant_mol=Molecule(smiles='[CH3]'))
        self.assertEqual(reversed_xyz_1, xyz)

        expected_xyz_2 = {'symbols': ('C', 'H', 'H', 'H', 'S', 'O', 'N'),
               'isotopes': (12, 1, 1, 1, 32, 16, 14),
               'coords': ((-0.49011438, -1.0370438, 0.15365747),
                          (-0.6486995, -1.85796321, -0.54773423),
                          (-0.60359153, -1.37304859, 1.18613964),
                          (-1.43009127, 0.23517346, -1.11797908),
                          (1.02558264, -0.04344404, -0.07343859),
                          (-0.25448248, 1.10710477, 0.18359696),
                          (-1.30762173, 0.15796567, -0.1048929))}
        reversed_xyz_2 = reverse_xyz(xyz=xyz, reactants_reversed=True, rmg_reactant_mol=Molecule(smiles='[CH3]'))
        self.assertEqual(reversed_xyz_2, expected_xyz_2)

    def test_find_distant_neighbor(self):
        """Test the find_distant_neighbor() function."""
        xyz_1 = {'symbols': ('S', 'O', 'O', 'N', 'C', 'H', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 16, 16, 14, 12, 1, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433),
                            (-1.34135876, 1.49608206, 0.53295071))}
        mol_1 = ARCSpecies(label='CS(=O)(O)[NH]', xyz=xyz_1).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=9), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_1, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=2), [1, 3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=4), [1, 2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=3), [1, 2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_1, start=0), [5, 6, 7, 8, 9])

        xyz_2 = {'symbols': ('S', 'H', 'O', 'N', 'C', 'H', 'H', 'H', 'H'),
                 'isotopes': (32, 1, 16, 14, 12, 1, 1, 1, 1),
                 'coords': ((-0.06618943, -0.12360663, -0.07631983),
                            (-0.79539707, 0.86755487, 1.02675668),
                            (-0.68919931, 0.25421823, -1.34830853),
                            (0.01546439, -1.54297548, 0.44580391),
                            (1.59721519, 0.47861334, 0.00711),
                            (1.94428095, 0.40772394, 1.03719428),
                            (2.20318015, -0.14715186, -0.64755729),
                            (1.59252246, 1.5117895, -0.33908352),
                            (-0.8785689, -2.02453514, 0.38494433))}
        mol_2 = ARCSpecies(label='C[SH](=O)[NH]', xyz=xyz_2).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=8), 0)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_2, start=5), 0)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=2), [3, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=4), [2, 3])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=3), [2, 4])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_2, start=0), [5, 6, 7, 8])

        xyz_3 = """ C                 -2.27234259   -0.78101274   -0.00989219
 H                 -1.94047502   -1.78971513   -0.14135849
 H                 -1.87026865   -0.16873290   -0.78986011
 H                 -3.34123881   -0.74952096   -0.04689398
 C                 -1.79025824   -0.25578524    1.35514655
 H                 -2.12212581    0.75291715    1.48661285
 H                 -2.19233218   -0.86806508    2.13511447
 N                 -0.32177464   -0.29904964    1.40598078
 H                  0.05399540    0.27317450    0.67703880
 H                 -0.00873286    0.04200717    2.29236958"""
        mol_3 = ARCSpecies(label='EA', smiles='NCC', xyz=xyz_3).mol
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=9), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=8), 4)
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=5), [0, 7])
        self.assertIn(find_distant_neighbor(rmg_mol=mol_3, start=6), [0, 7])
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=1), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)
        self.assertEqual(find_distant_neighbor(rmg_mol=mol_3, start=2), 4)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(ARC_PATH, 'arc', 'testing', 'heuristics'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
