#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.scheduler module
"""

import datetime
import pprint
import unittest
import os
import shutil

import numpy as np

import arc.checks as checks
import arc.rmgdb as rmgdb
import arc.parser as parser
from arc.common import ARC_PATH, almost_equal_coords_lists
from arc.job.factory import job_factory
from arc.level import Level
from arc.plotter import save_conformers_file
from arc.scheduler import Scheduler
from arc.imports import settings
from arc.parser import parse_normal_mode_displacement
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies, TSGuess


default_levels_of_theory = settings['default_levels_of_theory']


class TestChecks(unittest.TestCase):
    """
    Contains unit tests for the check module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)
        cls.rms_list_1 = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026, 0.08944271909999159,
                          0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        path_1 = os.path.join(checks.ARC_PATH, 'arc', 'testing', 'freq', 'C3H7_intra_h_TS.out')
        cls.freqs_1, cls.normal_modes_disp_1 = parse_normal_mode_displacement(path_1)
        cls.ts_1 = ARCSpecies(label='TS', is_ts=True)
        cls.ts_1.ts_guesses = [TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               ]

    def test_sum_time_delta(self):
        """Test the sum_time_delta() function"""
        dt1 = datetime.timedelta(days=0, minutes=0, seconds=0)
        dt2 = datetime.timedelta(days=0, minutes=0, seconds=0)
        dt3 = datetime.timedelta(days=0, minutes=1, seconds=15)
        dt4 = datetime.timedelta(days=10, minutes=1, seconds=15, microseconds=300)
        self.assertEqual(checks.sum_time_delta([]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(checks.sum_time_delta([dt1]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(checks.sum_time_delta([dt1, dt2]), datetime.timedelta(days=0, minutes=0, seconds=0))
        self.assertEqual(checks.sum_time_delta([dt1, dt3]), datetime.timedelta(days=0, minutes=1, seconds=15))
        self.assertEqual(checks.sum_time_delta([dt3, dt4]), datetime.timedelta(days=10, minutes=2, seconds=30, microseconds=300))

    def test_get_i_from_job_name(self):
        """Test the get_i_from_job_name() function"""
        self.assertIsNone(checks.get_i_from_job_name(''))
        self.assertIsNone(checks.get_i_from_job_name('some_job_name'))
        self.assertEqual(checks.get_i_from_job_name('conformer3'), 3)
        self.assertEqual(checks.get_i_from_job_name('conformer33'), 33)
        self.assertEqual(checks.get_i_from_job_name('conformer3355'), 3355)
        self.assertEqual(checks.get_i_from_job_name('tsg2'), 2)

    def test_check_ts_freq_job(self):
        """Test the check_ts_freq_job() function"""
        ts_xyz = """C       0.52123900   -0.93806900   -0.55301700
C       0.15387500    0.18173100    0.37122900
C      -0.89554000    1.16840700   -0.01362800
H       0.33997700    0.06424800    1.44287100
H       1.49602200   -1.37860200   -0.29763200
H       0.57221700   -0.59290500   -1.59850500
H       0.39006800    1.39857900   -0.01389600
H      -0.23302200   -1.74751100   -0.52205400
H      -1.43670700    1.71248300    0.76258900
H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
        r_xyz = """C                  0.50180491   -0.93942231   -0.57086745
 C                  0.01278145    0.13148427    0.42191407
 C                 -0.86874485    1.29377369   -0.07163907
 H                  0.28549447    0.06799101    1.45462711
 H                  1.44553946   -1.32386345   -0.24456986
 H                  0.61096295   -0.50262210   -1.54153222
 H                 -0.24653265    2.11136864   -0.37045418
 H                 -0.21131163   -1.73585284   -0.61629002
 H                 -1.51770930    1.60958621    0.71830245
 H                 -1.45448167    0.96793094   -0.90568876"""
        p_xyz = """C                  0.48818717   -0.94549701   -0.55196729
 C                  0.35993708    0.29146456    0.35637075
 C                 -0.91834764    1.06777042   -0.01096751
 H                  0.30640232   -0.02058840    1.37845537
 H                  1.37634603   -1.48487836   -0.29673876
 H                  0.54172192   -0.63344406   -1.57405191
 H                  1.21252186    0.92358349    0.22063264
 H                 -0.36439762   -1.57761595   -0.41622918
 H                 -1.43807526    1.62776079    0.73816131
 H                 -1.28677889    1.04716138   -1.01532486"""
        ts_spc = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz)
        ts_spc.mol_from_xyz()
        reactant = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=r_xyz)
        product = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=p_xyz)
        reactant.e0, product.e0, ts_spc.e0 = 0, 10, 100
        rxn = ARCReaction(r_species=[reactant], p_species=[product])
        rxn.ts_species = ts_spc
        rxn.determine_family(self.rmgdb)
        job = job_factory(job_adapter='gaussian',
                          species=[ts_spc],
                          job_type='composite',
                          level=Level(method='CBS-QB3'),
                          project='test_project',
                          project_directory=os.path.join(ARC_PATH,
                                                         'Projects',
                                                         'arc_project_for_testing_delete_after_usage4'),
                          )
        job.local_path_to_output_file = os.path.join(checks.ARC_PATH, 'arc', 'testing', 'composite',
                                                     'TS_intra_H_migration_CBS-QB3.out')
        print(ts_spc.ts_checks)
        switch_ts = checks.check_ts_freq_job(species=ts_spc, reaction=rxn, job=job)
        print(switch_ts)
        print(ts_spc.ts_checks)
        raise



    def test_get_indices_of_atoms_participating_in_reaction(self):
        """Test the get_indices_of_atoms_participating_in_reaction() function"""
        self.assertEqual(checks.get_indices_of_atoms_participating_in_reaction(normal_mode_disp=self.normal_modes_disp_1,
                                                                               freqs=self.freqs_1,
                                                                               ts_guesses=self.ts_1.ts_guesses,
                                                                               ), [3, 0, 1])

    def test_invalidate_rotors_with_both_pivots_in_a_reactive_zone(self):
        """Test the invalidate_rotors_with_both_pivots_in_a_reactive_zone() function"""
        ts_xyz_1 = """O      -0.63023600    0.92494700    0.43958200
        C       0.14513500   -0.07880000   -0.04196400
        C      -0.97050300   -1.02992900   -1.65916600
        N      -0.75664700   -2.16458700   -1.81286400
        H      -1.25079800    0.57954500    1.08412300
        H       0.98208300    0.28882200   -0.62114100
        H       0.30969500   -0.94370100    0.59100600
        H      -1.47626400   -0.10694600   -1.88883800"""  # 'N#[CH].[CH2][OH]'
        ts_spc_1 = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz_1)
        ts_spc_1.mol_from_xyz()
        ts_spc_1.determine_rotors()
        # Add a rotor that is breaks the TS and is not identified automatically.
        ts_spc_1.rotors_dict[1] = {'pivots': [2, 3],
                                  'top': [4, 8],
                                  'scan': [1, 2, 3, 4],
                                  'torsion': [0, 1, 2, 3],
                                  'success': None,
                                  'invalidation_reason': '',
                                  'dimensions': 1}
        rxn_zone_atom_indices = [1, 2]
        checks.invalidate_rotors_with_both_pivots_in_a_reactive_zone(species=ts_spc_1,
                                                                     rxn_zone_atom_indices=rxn_zone_atom_indices)
        self.assertEqual(ts_spc_1.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(ts_spc_1.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(ts_spc_1.rotors_dict[0]['success'])
        self.assertEqual(ts_spc_1.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(ts_spc_1.rotors_dict[1]['scan'], [1, 2, 3, 4])
        self.assertEqual(ts_spc_1.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(ts_spc_1.rotors_dict[1]['success'], False)

        ts_xyz_2 = """C       0.52123900   -0.93806900   -0.55301700
C       0.15387500    0.18173100    0.37122900
C      -0.89554000    1.16840700   -0.01362800
H       0.33997700    0.06424800    1.44287100
H       1.49602200   -1.37860200   -0.29763200
H       0.57221700   -0.59290500   -1.59850500
H       0.39006800    1.39857900   -0.01389600
H      -0.23302200   -1.74751100   -0.52205400
H      -1.43670700    1.71248300    0.76258900
H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
        ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=ts_xyz_2)
        ts_spc_2.mol_from_xyz()
        rxn_zone_atom_indices = [1, 2, 6]
        checks.invalidate_rotors_with_both_pivots_in_a_reactive_zone(species=ts_spc_2,
                                                                     rxn_zone_atom_indices=rxn_zone_atom_indices)
        self.assertEqual(ts_spc_2.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(ts_spc_2.rotors_dict[0]['scan'], [5, 1, 2, 3])
        self.assertEqual(ts_spc_2.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(ts_spc_2.rotors_dict[0]['success'])
        self.assertEqual(ts_spc_2.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(ts_spc_2.rotors_dict[1]['scan'], [1, 2, 3, 9])
        self.assertEqual(ts_spc_2.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(ts_spc_2.rotors_dict[1]['success'], False)

    def test_get_index_of_abs_largest_neg_freq(self):
        """Test the get_index_of_abs_largest_neg_freq() function"""
        self.assertIsNone(checks.get_index_of_abs_largest_neg_freq(np.array([], np.float64)))
        self.assertIsNone(checks.get_index_of_abs_largest_neg_freq(np.array([1, 320.5], np.float64)))
        self.assertEqual(checks.get_index_of_abs_largest_neg_freq(np.array([-1], np.float64)), 0)
        self.assertEqual(checks.get_index_of_abs_largest_neg_freq(np.array([-1, 320.5], np.float64)), 0)
        self.assertEqual(checks.get_index_of_abs_largest_neg_freq(np.array([320.5, -1], np.float64)), 1)
        self.assertEqual(checks.get_index_of_abs_largest_neg_freq(np.array([320.5, -1, -80, -90, 5000], np.float64)), 3)
        self.assertEqual(checks.get_index_of_abs_largest_neg_freq(np.array([-320.5, -1, -80, -90, 5000], np.float64)), 0)

    def test_get_expected_num_atoms_with_largest_normal_mode_disp(self):
        """Test the get_expected_num_atoms_with_largest_normal_mode_disp() function"""
        normal_disp_mode_rms = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026,
                                0.08944271909999159, 0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        num_of_atoms = checks.get_expected_num_atoms_with_largest_normal_mode_disp(
            normal_disp_mode_rms=normal_disp_mode_rms,
            ts_guesses=self.ts_1.ts_guesses)
        self.assertEqual(num_of_atoms, 4)

    def test_get_rxn_normal_mode_disp_atom_number(self):
        """Test the get_rxn_normal_mode_disp_atom_number function"""
        self.assertEqual(checks.get_rxn_normal_mode_disp_atom_number('default'), 3)
        self.assertEqual(checks.get_rxn_normal_mode_disp_atom_number('intra_H_migration'), 3)
        self.assertEqual(checks.get_rxn_normal_mode_disp_atom_number('intra_H_migration', rms_list=self.rms_list_1), 4)

    def test_get_rms_from_normal_modes_disp(self):
        """Test the get_rms_from_normal_modes_disp() function"""
        rms = checks.get_rms_from_normal_mode_disp(self.normal_modes_disp_1, np.array([-1000.3, 320.5], np.float64))
        self.assertEqual(rms, [0.07874007874011811,
                               0.07280109889280519,
                               0.0,
                               0.9914635646356349,
                               0.03605551275463989,
                               0.034641016151377546,
                               0.0,
                               0.033166247903554,
                               0.01414213562373095,
                               0.0],
                         )

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage4']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
