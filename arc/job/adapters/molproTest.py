#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.molpro module
"""

import math
import os
import shutil
import unittest

from arc.job.adapters.gaussian import GaussianAdapter
from arc.level import Level
from arc.settings import (arc_path,
                          default_job_settings,
                          global_ess_settings,
                          input_filenames,
                          output_filenames,
                          rotor_scan_resolution,
                          servers,
                          submit_filenames,
                          )
from arc.species import ARCSpecies


class TestGaussianAdapter(unittest.TestCase):
    """
    Contains unit tests for the GaussianAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = GaussianAdapter(execution_type='incore',
                                    job_type='composite',
                                    level=Level(method='cbs-qb3-paraskevas'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    args={'keyword': {'general': 'IOp(1/12=5,3/44=0)'}},
                                    )
        cls.job_2 = GaussianAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1']),
                                             ARCSpecies(label='spc2', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.job_3 = GaussianAdapter(execution_type='queue',
                                    job_type='opt',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP',
                                                solvation_method='SMD',
                                                solvent='Water'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='spc1', xyz=['O 0 0 1'])],
                                    testing=True,
                                    )
        cls.job_4 = GaussianAdapter(execution_type='queue',
                                    job_type='scan',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='Argon', xyz=['Ar 0 0 0'])],
                                    testing=True,
                                    scan=[[1, 2, 3, 4], [4, 5, 6, 7]],
                                    args={'block': {'general': 'additional\ngaussian\nblock'}},
                                    )
        cls.job_5 = GaussianAdapter(execution_type='queue',
                                    job_type='freq',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='birad singlet',
                                                        xyz=['O 0 0 1'],
                                                        multiplicity=1,
                                                        number_of_radicals=2)],
                                    testing=True,
                                    )
        cls.job_6 = GaussianAdapter(execution_type='queue',
                                    job_type='optfreq',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='anion TS', xyz=['O 0 0 1'], charge=-1, is_ts=True)],
                                    testing=True,
                                    )
        cls.job_7 = GaussianAdapter(execution_type='queue',
                                    job_type='irc',
                                    level=Level(method='wb97xd',
                                                basis='def2-TZVP'),
                                    project='test',
                                    project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'),
                                    species=[ARCSpecies(label='IRC', xyz=['O 0 0 1'], is_ts=True)],
                                    irc_direction='reverse',
                                    testing=True,
                                    )

    def test_set_cpu_and_mem(self):
        """Test assigning number of cpu's and memory"""
        self.job_1.cpu_cores = 8
        self.job_1.input_file_memory = None
        self.job_1.submit_script_memory = None
        self.job_1.server = 'server2'
        self.job_1.software = 'gaussian'
        self.job_1.set_cpu_and_mem()
        self.assertEqual(self.job_1.cpu_cores, 8)
        expected_memory = math.ceil(14 * 1024)
        self.assertEqual(self.job_1.input_file_memory, expected_memory)

    def test_set_input_file_memory(self):
        """Test setting the input_file_memory argument"""
        self.assertEqual(self.job_1.input_file_memory, self.job_1.job_memory_gb * 1024)
        self.assertEqual(self.job_2.input_file_memory, 14336)

    def test_write_input_file(self):
        """Test writing Gaussian input files"""
        self.job_1.write_input_file()
        with open(os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]), 'r') as f:
            content_1 = f.read()
        job_1_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc) guess=mix cbs-qb3   IOp(2/9=2000) IOp(1/12=5,3/44=0) scf=xqc  

spc1

0 3
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_1, job_1_expected_input_file)

        self.job_3.write_input_file()
        with open(os.path.join(self.job_3.local_path, input_filenames[self.job_3.job_adapter]), 'r') as f:
            content_3 = f.read()
        job_3_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(calcfc) SCRF=(smd, Solvent=water) guess=mix uwb97xd/def2-tzvp   IOp(2/9=2000) scf=xqc  

spc1

0 3
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_3, job_3_expected_input_file)

        self.job_4.write_input_file()
        with open(os.path.join(self.job_4.local_path, input_filenames[self.job_4.job_adapter]), 'r') as f:
            content_4 = f.read()
        job_4_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(modredundant, calcfc, noeigentest, maxStep=5) scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12) guess=mix wb97xd/def2-tzvp   IOp(2/9=2000) scf=xqc  

Argon

0 1
Ar      0.00000000    0.00000000    0.00000000

D 1 2 3 4 S 45 8.0
D 4 5 6 7 S 45 8.0


additional
gaussian
block



"""
        self.assertEqual(content_4, job_4_expected_input_file)

        self.job_5.write_input_file()
        with open(os.path.join(self.job_5.local_path, input_filenames[self.job_5.job_adapter]), 'r') as f:
            content_5 = f.read()
        job_5_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P  guess=mix uwb97xd/def2-tzvp freq IOp(7/33=1) scf=(tight, direct) integral=(grid=ultrafine, Acc2E=12)  IOp(2/9=2000) scf=xqc  

birad singlet

0 1
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_5, job_5_expected_input_file)

        self.job_6.write_input_file()
        with open(os.path.join(self.job_6.local_path, input_filenames[self.job_6.job_adapter]), 'r') as f:
            content_6 = f.read()
        job_6_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P opt=(ts, calcfc, noeigentest, maxcycles=100) guess=mix uwb97xd/def2-tzvp   IOp(2/9=2000) scf=xqc  

anion TS

-1 2
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_6, job_6_expected_input_file)

        self.job_7.write_input_file()
        with open(os.path.join(self.job_7.local_path, input_filenames[self.job_7.job_adapter]), 'r') as f:
            content_7 = f.read()
        job_7_expected_input_file = """%chk=check.chk
%mem=14336mb
%NProcShared=8

#P irc=(CalcAll, reverse, maxpoints=50, stepsize=7) guess=mix wb97xd/def2-tzvp   IOp(2/9=2000) scf=xqc  

IRC

0 1
O       0.00000000    0.00000000    1.00000000


"""
        self.assertEqual(content_7, job_7_expected_input_file)

    def test_set_up_files(self):
        """Test setting up files"""
        job_1_files_to_upload = [{'file_name': 'input.gjf',
                                  'local': os.path.join(self.job_1.local_path, input_filenames[self.job_1.job_adapter]),
                                  'remote': os.path.join(self.job_1.remote_path, input_filenames[self.job_1.job_adapter]),
                                  'source': 'path',
                                  'make_x': False}]
        job_1_files_to_download = [{'file_name': 'input.log',
                                    'local': os.path.join(self.job_1.local_path, output_filenames[self.job_1.job_adapter]),
                                    'remote': os.path.join(self.job_1.remote_path, output_filenames[self.job_1.job_adapter]),
                                    'source': 'path',
                                    'make_x': False},
                                   {'file_name': 'check.chk',
                                    'local': os.path.join(self.job_1.local_path, 'check.chk'),
                                    'remote': os.path.join(self.job_1.remote_path, 'check.chk'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_1.files_to_upload, job_1_files_to_upload)
        self.assertEqual(self.job_1.files_to_download, job_1_files_to_download)

        job_2_files_to_upload = [{'file_name': 'submit.sh',
                                  'local': os.path.join(self.job_2.local_path, 'submit.sh'),
                                  'remote': os.path.join(self.job_2.remote_path, 'submit.sh'),
                                  'source': 'path',
                                  'make_x': False},
                                 {'file_name': 'data.hdf5',
                                  'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
                                  'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
                                  'source': 'path',
                                  'make_x': False}]
        job_2_files_to_download = [{'file_name': 'data.hdf5',
                                    'local': os.path.join(self.job_2.local_path, 'data.hdf5'),
                                    'remote': os.path.join(self.job_2.remote_path, 'data.hdf5'),
                                    'source': 'path',
                                    'make_x': False}]
        self.assertEqual(self.job_2.files_to_upload, job_2_files_to_upload)
        self.assertEqual(self.job_2.files_to_download, job_2_files_to_download)


    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        shutil.rmtree(os.path.join(arc_path, 'arc', 'testing', 'test_GaussianAdapter'))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
