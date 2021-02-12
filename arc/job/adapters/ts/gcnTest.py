#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.gcn module
"""

import os
import shutil
import unittest

import arc.rmgdb as rmgdb
from arc.common import arc_path
from arc.job.adapters.ts.gcn_ts import GCNAdapter
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies, TSGuess


class TestGCNAdapter(unittest.TestCase):
    """
    Contains unit tests for the GCNAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)
        cls.output_dir = os.path.join(arc_path, 'arc', 'testing', 'GCN')
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)

    def test_gcn(self):
        """
        Test that ARC can call GCN to make TS guesses for further optimization.
        """
        reactant_xyz = """C  -1.3087    0.0068    0.0318
        C  0.1715   -0.0344    0.0210
        N  0.9054   -0.9001    0.6395
        O  2.1683   -0.5483    0.3437
        N  2.1499    0.5449   -0.4631
        N  0.9613    0.8655   -0.6660
        H  -1.6558    0.9505    0.4530
        H  -1.6934   -0.0680   -0.9854
        H  -1.6986   -0.8169    0.6255"""
        reactant = ARCSpecies(label='reactant', smiles='C([C]1=[N]O[N]=[N]1)', xyz=reactant_xyz)

        product_xyz = """C  -1.0108   -0.0114   -0.0610  
        C  0.4780    0.0191    0.0139    
        N  1.2974   -0.9930    0.4693    
        O  0.6928   -1.9845    0.8337    
        N  1.7456    1.9701   -0.6976    
        N  1.1642    1.0763   -0.3716    
        H  -1.4020    0.9134   -0.4821  
        H  -1.3327   -0.8499   -0.6803   
        H  -1.4329   -0.1554    0.9349"""
        product = ARCSpecies(label='product', smiles='[N-]=[N+]=C(N=O)C', xyz=product_xyz)

        rxn1 = ARCReaction(label='reactant <=> product', ts_label='TS0')
        rxn1.r_species = [reactant]
        rxn1.p_species = [product]

        gcn1 = GCNAdapter(job_type='tsg',
                          reactions=[rxn1],
                          testing=True,
                          project='test',
                          project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GCN'))
        gcn1.local_path = self.output_dir
        gcn1.execute_incore()

        self.assertEqual(rxn1.ts_species.multiplicity, 1)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 2)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'))
        self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 9)
        self.assertTrue(rxn1.ts_species.ts_guesses[0].success)
        self.assertTrue(rxn1.ts_species.ts_guesses[0].execution_time.seconds < 59)  # 0:00:01.985336

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(arc_path, 'arc', 'testing', 'test_GCN'))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
