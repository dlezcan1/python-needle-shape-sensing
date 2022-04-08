"""
Created on Aug 3, 2020

This is a class file for FBG Needle parameterizations

@author: Dimitri Lezcano
"""
import argparse
import json
import os
from itertools import product
from typing import Union
from warnings import warn

import numpy as np

from . import fbg_signal_processing


class Needle( object ):
    """ basic Needle class """
    # Needle Diameters (mm)
    DIAM_17G = 1.473
    DIAM_18G = 1.27
    DIAM_19G = 1.067
    DIAM_20G = 0.908
    DIAM_21G = 0.819
    DIAM_GAUGE = { '17G': DIAM_17G,
                   '18G': DIAM_18G,
                   '19G': DIAM_19G,
                   '20G': DIAM_20G,
                   '21G': DIAM_21G,
                   }

    # Young's Modulus (N/mm^2)
    EMOD_ST_STEEL304 = 200 * 10 ** 3  # N/mm^2
    EMOD_NITINOL = 83 * 10 ** 3  # N/mm^2
    EMOD = { 'stainless-steel-304': EMOD_ST_STEEL304,
             'nitinol'            : EMOD_NITINOL
             }

    # Poisson's Ratio
    P_RATIO_ST_STEEL304 = 0.29
    P_RATIO_NITINOL = 0.33
    P_RATIO = { 'stainless-steel-304': P_RATIO_ST_STEEL304,
                'nitinol'            : P_RATIO_NITINOL
                }

    def __init__( self, length: float, serial_number: str, diameter: float = None, Emod: float = None,
                  pratio: float = None ):
        """ constructor

            Args:
                - length: float, of the length of the entire needle (mm)
                - serial_number: the Serial number of the needle
                - diameter: A float of the diameter
                - Emod: Young's modulus
                - pratio: Poisson's Ratio of the material

        """
        self._length = length
        self._serial_number = serial_number
        self.diameter = diameter
        self.Emod = Emod
        self.pratio = pratio

    # __init__

    def __str__( self ):
        msg = f"Serial Number: {self.serial_number}"
        msg += "\n" + f"Needle length (mm): {self.length}"
        msg += "\n" + f"Diameter (mm): {self.diameter}" if self.diameter is not None else ""
        msg += "\n" + f"Bending Moment of Insertia (mm^4): {self.I_bend}" if self.I_bend is not None else ""
        msg += "\n" + f"Torsional Moment of Insertia (mm^4): {self.J_torsion}" if self.J_torsion is not None else ""
        msg += "\n" + f"Young's Modulus, Emod (N/mm^2): {self.Emod}" if self.Emod is not None else ""
        msg += "\n" + f"Torsional Young's Modulus, Gmod (N/mm^2): {self.Gmod}" if self.Gmod is not None else ""

        return msg

    # __str__

    # =================== PROPERTIES ============================= #
    @property
    def B( self ):
        """ The Stiffness matrix of the needle """
        if self.Emod > 0 and self.diameter > 0 and self.pratio > 0:
            return np.diag( [ self.bend_stiffness, self.bend_stiffness, self.torsional_stiffness ] )

        else:
            return None

    # B

    @property
    def bend_stiffness( self ):
        """ The bending stiffness of the needle"""
        if self.Emod is not None and self.I_bend is not None:
            return self.Emod * self.I_bend

        else:
            return None

    # bend_stiffness

    @property
    def I_bend( self ):
        """ Bending moment of insetia"""
        if self.diameter is not None:
            return np.pi * self.diameter ** 4 / 64

        else:
            return None

    # I_bend

    @property
    def J_torsion( self ):
        """ Torsional moment of inertia"""
        if self.diameter is not None:
            return np.pi * self.diameter ** 4 / 32

        else:
            return None

    # J_torsion

    @property
    def Gmod( self ):
        """ Torsional Young's Modulus"""
        if self.Emod is not None and self.pratio is not None:
            return self.Emod / (2 * (1 + self.pratio))

        else:
            return None

    # Gmod

    @property
    def length( self ):
        return self._length

    # length

    @property
    def serial_number( self ):
        return self._serial_number

    # serial_number

    @property
    def torsional_stiffness( self ):
        """ Torsional stiffness of the needle"""
        if self.Gmod is not None and self.J_torsion is not None:
            return self.Gmod * self.J_torsion

        else:
            return None

    # torsional_stiffness

    # =============== FUNCTIONS ================================== #
    def to_dict( self ) -> dict:
        """ Convert object to a dictionary"""
        return { 'serial number': self.serial_number,
                 'length'       : self.length,
                 'diameter'     : self.diameter,
                 'Emod'         : self.Emod,
                 'pratio'       : self.pratio
                 }

    # to_dict


# class: Needle


class FBGNeedle( Needle ):
    """
    This is a class for FBG Needle parameters containment.
    """

    def __init__( self, length: float, serial_number: str, num_channels: int, sensor_location=None,
                  calibration_mats=None, weights=None, **kwargs ):
        """
        Constructor

        Args:
            - length: float, of the length of the entire needle (mm)
            - num_channels: int, the number of channels there are
            - sensor_location: list, the arclength locations (mm) of the AA's (default = None)
                This measurement is from the base of the needle
        """

        # data checking
        if sensor_location is None:
            sensor_location = [ ]

        if weights is None:
            weights = { }

        if calibration_mats is None:
            calibration_mats = { }

        if num_channels <= 0:
            raise ValueError( "'num_channels' must be > 0." )

        # if

        if len( sensor_location ) > 0:
            sensor_loc_invalid = [ loc > length or loc < 0 for loc in sensor_location ]
            if any( sensor_loc_invalid ):
                raise ValueError( "all sensor locations must be in [0, 'length']" )

        # if

        super().__init__( length, serial_number, **kwargs )

        # property set-up (None so that they are set once)
        self._num_channels = num_channels
        self._sensor_location = list( sensor_location )
        self._cal_matrices = { }
        self._weights = { }

        # assignments
        self.cal_matrices = calibration_mats
        self.weights = weights
        self.ref_wavelengths = -np.ones( self.num_channels * self.num_activeAreas )  # reference wavelengths

        # (static/class) methods -> instance methods
        self.calculate_length_measured = self.__calculate_length_measured
        self.generate_ch_aa = self.__generate_ch_aa
        self.assignments_aa = self.__assignments_aa
        self.assignments_ch = self.__assignments_ch

    # __init__

    def __str__( self ):
        """ Magic str method """
        msg = super().__str__()
        msg += "\nNumber of FBG Channels: {:d}".format( self.num_channels )
        msg += "\nNumber of Active Areas: {:d}".format( self.num_activeAreas )
        msg += "\nSensor Locations (mm):"
        if self.num_activeAreas > 0:
            for i in range( self.num_activeAreas ):
                msg += "\n\t{:d}: {}".format( i + 1, self.sensor_location[ i ] )

            # for
        # if
        else:
            msg += " None"

        # else

        if self.cal_matrices:
            msg += "\nCalibration Matrices:"
            for loc, cal_mat in self.cal_matrices.items():
                msg += "\n\t{}: ".format( loc ) + str( cal_mat.tolist() )

                if self.weights:
                    msg += " | weight: " + str( self.weights[ loc ] )

                # if
            # for
        # if

        return msg

    # __str__

    def __repr__( self ):
        return "FBGneedle:\n" + str( self )

    # __repr__

    ############################## PROPERTIES ######################################
    @property
    def is_calibrated( self ):
        return (self.cal_matrices is not None) and (len( self._cal_matrices ) > 0)

    # is_calibrated

    @property
    def num_aa( self ):
        DeprecationWarning( 'num_aa is deprecated. Please use num_activeAreas.' )
        return len( self.sensor_location )

    # num_aa

    @property
    def num_activeAreas( self ):
        return len( self.sensor_location )

    # num_activeAreas

    @property
    def num_channels( self ):
        return self._num_channels

    # num_channels

    @property
    def num_signals( self ):
        return self.num_channels * self.num_activeAreas

    # num_signals

    @property
    def sensor_calibrated( self ):
        return all( self.ref_wavelengths >= 0 )

    # sensor_calibrated

    @property
    def sensor_location( self ):
        return self._sensor_location

    # sensor_locations

    @property
    def sensor_location_tip( self ):
        return [ self.length - base_loc for base_loc in self.sensor_location ]

    # sensor_location_tip

    @property
    def cal_matrices( self ):
        return self._cal_matrices

    # cal_matrices

    @cal_matrices.setter
    def cal_matrices( self, C_dict: dict ):
        for key, C in C_dict.items():
            # get the sensor location
            # check of 'AAX' format
            if isinstance( key, str ):
                loc = self.aa_loc( key )

            # if

            elif isinstance( key, int ):
                # check if it is alrady a sensor location
                if key in self.sensor_location:
                    loc = key

                # if

                # if not, check to see if it is an AA index [1, #AA]
                elif key in range( 1, self.num_activeAreas + 1 ):
                    loc = self.sensor_location[ key - 1 ]

                # elif

                else:
                    raise KeyError( "Sensor location is not a valid sensor location" )

                # else

            # elif

            elif isinstance( key, float ):
                if key in self.sensor_location:
                    loc = key

                # if

                else:
                    raise KeyError( "Sensor location is not a valid sensor location." )

                # else

            # elif
            else:
                raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

            # else

            self._cal_matrices[ loc ] = C

        # for          

    # cal_matrices: setter

    @property
    def weights( self ):
        return self._weights

    # weights

    @weights.setter
    def weights( self, weights: dict ):
        for key, weight in weights.items():
            # get the sensor location
            if isinstance( key, str ):
                loc = self.aa_loc( key )

            # if

            elif isinstance( key, int ):
                # check if it is alrady a sensor location
                if key in self.sensor_location:
                    loc = key

                # if

                # if not, check to see if it is an AA index [1, #AA]
                elif key in range( 1, self.num_activeAreas + 1 ):
                    loc = self.sensor_location[ key - 1 ]

                # elif

                else:
                    raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

                # else

            # elif

            elif isinstance( key, float ):
                if key in self.sensor_location:
                    loc = key

                # if

                else:
                    raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

                # else

            else:
                raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

            # else

            # add the weight to weights
            self._weights[ loc ] = weight

        # for

    # weights setter

    ######################## FUNCTIONS ######################################

    def aa_cal( self, aax: str ):
        """ Function to get calibration matrix from AAX indexing """
        return self.cal_matrices[ self.aa_loc( aax ) ]

    # aa_cal

    @staticmethod
    def aa_idx( aax: str ):
        """ Function to get value from AAX indexing """
        return int( "".join( filter( str.isdigit, aax ) ) ) - 1

    # get_aa

    def aa_loc( self, aax: str ):
        """ Function to get location from AAX indexing """
        return self.sensor_location[ self.aa_idx( aax ) ]

    # aa_loc

    @staticmethod
    def assignments_aa( num_channels: int, num_active_areas: int ) -> list:
        """ Function for returning a list of AA assignments """
        return list( range( 1, num_active_areas + 1 ) ) * num_channels

    # assignments_aa

    def __assignments_aa( self ) -> list:
        """ Instance method of assignments_aa """
        return FBGNeedle.assignments_aa( self.num_channels, self.num_activeAreas )

    # __assignments_aa

    @staticmethod
    def assignments_ch( num_channels: int, num_active_areas: int ) -> list:
        """ Function for returning a list of CH assignments"""
        return sum( [ num_active_areas * [ ch_i ] for ch_i in range( 1, num_channels + 1 ) ], [ ] )

    # assignments_ch

    def __assignments_ch( self ) -> list:
        """ Instance method of assignments_ch """
        return FBGNeedle.assignments_ch( self.num_channels, self.num_activeAreas )

    # __assignments_ch

    @staticmethod
    def calculate_length_measured( s_m: np.ndarray, L: float, tip: bool = True, needle_length: float = None,
                                   valid: bool = False ):
        """ Determine (and return) which lengths are valid

            :param s_m: the measured arclengths
            :param L:   the insertion depth
            :param tip: (Default = True) whether the measured arclengths are from the tip of the needle or not.
            :param needle_length: (Default = None) float of the entire needle length. Only needed if tip = False
            :param valid: (Default = False) whether to only return valid arclengths
        """
        # calculate the valid measurement locations
        if tip:
            s_m_valid = L - s_m

        elif needle_length is not None:
            s_m_valid = L - needle_length + s_m

        else:
            raise ValueError( "If you are using base calculations, you need to define a 'needle_length'" )

        # else

        # determine valid AA locations
        s_m_valid_mask = s_m_valid >= 0
        if valid:
            s_m_valid = s_m_valid[ s_m_valid_mask ]

        # if

        return s_m_valid, s_m_valid_mask

    # calculate_length_measured

    def __calculate_length_measured( self, L: float, tip: bool = True, valid: bool = False ):
        """ Determine (and return) which lengths are valid for the current FBGNeedle

            :param L:   the insertion depth
            :param tip: (Default = True) whether the measured arclengths are from the tip of the needle or not.
            :param valid: (Default = False) whether to only return valid arclengths
        """
        # nump-ify the sensor locations
        if tip:
            s_m = np.array( self.sensor_location_tip )
        else:
            s_m = np.array( self.sensor_location )

        return FBGNeedle.calculate_length_measured( s_m, L, tip=tip, needle_length=self.length, valid=valid )

    # __calculate_length_measured

    def curvatures_raw( self, raw_signals: Union[ dict, np.ndarray ], temp_comp: bool = True ) -> \
            Union[ dict, np.ndarray ]:
        """ Determine the curvatures from signals input

                    Args:
                        raw_signals: ({AA_index: signals} | numpy array of signals (can be multi-row))
                                      must be only the raw signals
                        temp_comp: bool (Default True) of whether to use temperature compensation

                    Return:
                        (dict of {AA_index: curvature_xy} | numpy array of size 2 x num_activeAreas of curvature_xy)
        """
        aa_assignments = self.assignments_aa()

        if isinstance( raw_signals, dict ):
            proc_signals = np.zeros( self.num_signals )
            raw_signals = { }
            for aa_i, raw_signal in raw_signals.items():
                # get the appropriate calibration matrix
                aa_idx = aa_i if isinstance( aa_i, int ) else int( "".join( filter( str.isdigit, aa_i ) ) )
                aa_i_mask = list( map( lambda aa: aa == aa_idx, aa_assignments ) )
                base_signal = self.ref_wavelengths[ aa_i_mask ]

                # process the signal
                proc_signal = fbg_signal_processing.process_signals( raw_signal, base_signal )
                proc_signals[ aa_i_mask ] = proc_signal

            # for

            # temperature compensate
            if temp_comp:
                # proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                #                                                                self.num_activeAreas )
                proc_signals = self.temperature_compensate(proc_signals, arg_check=False)

            # if

            curvatures = self.curvatures_processed( proc_signals )

        # if

        elif isinstance( raw_signals, np.ndarray ):
            # process the signals
            proc_signals = fbg_signal_processing.process_signals( raw_signals, self.ref_wavelengths )
            if temp_comp:
                # proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                #                                                                self.num_activeAreas )
                proc_signals = self.temperature_compensate( proc_signals, arg_check=False )

            # if

            if raw_signals.ndim == 1 and proc_signals.ndim == 2 and proc_signals.shape[ 0 ] == 1:
                proc_signals = proc_signals.squeeze( axis=0 )

            # if

            curvatures = self.curvatures_processed( proc_signals )

        # elif

        else:
            raise TypeError( "raw_signals must be a 'dict' or 'numpy.ndarray'" )

        # else

        return curvatures

    # curvatures_raw

    def curvatures_processed( self, proc_signals: Union[ dict, np.ndarray ] ) -> Union[ dict, np.ndarray ]:
        """ Determine the curvatures from signals input

            Args:
                proc_signals: {AA_index: processed signal} must be processed and temperature compensated

        """

        if isinstance( proc_signals, dict ):
            curvatures = { }
            for aa_i, proc_signal in proc_signals.items():
                # get the appropriate calibration matrix
                C_aa_i = self.aa_cal( f"AA{aa_i}" ) if isinstance( aa_i, int ) else self.aa_cal( aa_i )

                curvatures[ aa_i ] = C_aa_i @ proc_signal  # 2 x num_AA @ num_AA x 1

            # for

        # if

        elif isinstance( proc_signals, np.ndarray ):
            # initalize curvatures
            if proc_signals.ndim == 1:
                curvatures = np.zeros( (2, self.num_activeAreas) )
            elif proc_signals.ndim == 2:
                curvatures = np.zeros( (proc_signals.shape[ 0 ], 2, self.num_activeAreas) )
            else:
                raise IndexError( "'proc_signals' dimensions must be <= 2." )

            for aa_i in range( 1, self.num_activeAreas + 1 ):
                mask_aa_i = list( map( lambda aa: aa == aa_i, self.assignments_aa() ) )

                C_aa_i = self.aa_cal( f"AA{aa_i}" )

                if proc_signals.ndim == 1:
                    proc_signals_aa_i = proc_signals[ mask_aa_i ]
                    curvatures[ :, aa_i - 1 ] = C_aa_i @ proc_signals_aa_i

                # if

                else:
                    proc_signals_aa_i = proc_signals[ :, mask_aa_i ]
                    curvatures[ :, :, aa_i - 1 ] = proc_signals_aa_i @ C_aa_i.T

                # else
            # for

        # else

        else:
            raise TypeError( "'proc_signals' must be a 'dict' or 'numpy.ndarray'" )

        # else

        return curvatures

    # curvatures_processed

    @staticmethod
    def generate_ch_aa( num_channels: int, num_active_areas: int ) -> (list, list, list):
        """ Generate the CHX | AAY list

        """
        channels = [ f"CH{i}" for i in range( 1, num_channels + 1 ) ]
        active_areas = [ f"AA{i}" for i in range( 1, num_active_areas + 1 ) ]
        channel_active_area = [ " | ".join( (ch, aa) ) for ch, aa in product( channels, active_areas ) ]

        return channel_active_area, channels, active_areas

    # generate_ch_aa

    def __generate_ch_aa( self ) -> (list, list, list):
        """ Instance method of generate_ch_aa"""
        return FBGNeedle.generate_ch_aa( self.num_channels, self.num_activeAreas )

    # generate_chaa

    @staticmethod
    def load_json( filename: str ):
        """ 
        This function is used to load a FBGNeedle class from a saved JSON file.
        
        Args:
            - filename: str, the input json file to be loaded.
            
        Returns:
            A FBGNeedle Class object with the loaded json files.
        
        """
        # load the data from the json file to a dict
        with open( filename, 'r' ) as json_file:
            data = json.load( json_file )

        # with

        # insert the sensor locations in order of AA
        if 'Sensor Locations' in data.keys():
            sensor_locations = [ data[ 'Sensor Locations' ][ str( key ) ] for key in
                                 sorted( data[ 'Sensor Locations' ].keys(), ) ]

        # if

        else:
            sensor_locations = None

        # else

        # insert the calibration matrices
        if "Calibration Matrices" in data.keys():
            cal_mats = { }
            for loc, c_mat in data[ "Calibration Matrices" ].items():
                if isinstance( loc, str ):
                    loc = float( loc )
                cal_mats[ loc ] = np.array( c_mat )

            # for

        # if

        else:
            cal_mats = { }

        # else

        if "weights" in data.keys():
            weights = { }
            for loc, weight in data[ 'weights' ].items():
                if isinstance( loc, str ):
                    loc = float( loc )

                weights[ loc ] = float( weight )
                # for
        # if

        else:
            weights = { }

        # else

        # needle mechanical properties
        diameter = data.get( 'diameter', None )
        Emod = data.get( 'Emod', None )
        pratio = data.get( 'pratio', None )

        # instantiate the FBGNeedle class object
        fbg_needle = FBGNeedle( data[ 'length' ], data[ 'serial number' ], data[ '# channels' ], sensor_locations,
                                cal_mats, weights, diameter=diameter, Emod=Emod, pratio=pratio )

        # return the instantiation
        return fbg_needle

    # load_json

    def save_json( self, outfile: str = "needle_params.json" ):
        """
        This function is used to save the needle parameters as a JSON file.
        
        Args:
            - outfile: str, the output json file to be saved.
        
        """
        # place the saved data into the json file
        data = { "serial number" : self.serial_number, "length": self.length, "# channels": self.num_channels,
                 "# active areas": self.num_activeAreas }  # initialize the json dictionary

        if self.sensor_location:
            data[ "Sensor Locations" ] = { }
            for i, l in enumerate( self.sensor_location, 1 ):
                data[ "Sensor Locations" ][ str( i ) ] = l

            # for
        # if

        if self.cal_matrices:
            data[ "Calibration Matrices" ] = { }
            for k, cal_mat in self.cal_matrices.items():
                data[ "Calibration Matrices" ][ k ] = cal_mat.tolist()

            # for
        # if

        if self.weights:
            data[ 'weights' ] = { }
            for k, weight in self.weights.items():
                data[ 'weights' ][ k ] = weight

            # for
        # if

        data = self.to_dict()

        # write the data
        with open( outfile, 'w' ) as outfile:
            json.dump( data, outfile, indent=4 )

        # with

    # save_json

    def to_dict( self ) -> dict:
        """ Dictionary the values here """
        data = super().to_dict()
        data[ "# channels" ] = self.num_channels
        data[ "# active areas" ] = self.num_activeAreas

        if self.sensor_location:
            data[ "Sensor Locations" ] = { }
            for i, l in enumerate( self.sensor_location, 1 ):
                data[ "Sensor Locations" ][ str( i ) ] = l

            # for
        # if

        if self.cal_matrices:
            data[ "Calibration Matrices" ] = { }
            for k, cal_mat in self.cal_matrices.items():
                data[ "Calibration Matrices" ][ k ] = cal_mat.tolist()

            # for
        # if

        if self.weights:
            data[ 'weights' ] = { }
            for k, weight in self.weights.items():
                data[ 'weights' ][ k ] = weight

            # for
        # if

        return data

    # to_dict

    def set_calibration_matrices( self, cal_mats: dict ):
        """ This function is to set the calibration matrices after instantiation """
        warn( DeprecationWarning( "Use function property setter obj.cal_matrices = ..." ) )
        self.cal_matrices = cal_mats

    # set_calibration_matrices

    def set_weights( self, weights: dict ):
        """ This function is to set the weighting of their measurements """

        self.weights = weights

    # set_weights

    def temperature_compensate( self, proc_signals: np.ndarray, arg_check: bool = True ) -> np.ndarray:
        """ Perform temperature compensation for processed signals

            :param proc_signals: numpy array of processed signals (N x (# CHs x #AAs) size)
            :param arg_check: boolean of whether to check the input arguments or not

            :return: temperature compensated signals of the same size as proc_signals
        """
        # check size of signals
        if arg_check:
            if proc_signals.ndim == 1 and proc_signals.shape[ 0 ] != self.num_signals:
                raise AttributeError( "Size of processed signals is incorrect." )

            elif proc_signals.ndim == 2 and proc_signals.shape[ 1 ] != self.num_signals:
                raise AttributeError( "Size of processed signals is incorrect." )

            elif proc_signals.ndim > 2 or proc_signals.ndim < 1:
                raise AttributeError( "Size of processed signals must be a 1D vector or 2D matrix." )

        # if

        # get AA assignments
        aa_assignments = np.array( self.__assignments_aa() )
        num_dims = proc_signals.ndim
        proc_signals_Tcomp = proc_signals.copy()


        # iterate through active areas
        for aa_i in range( 1, self.num_activeAreas + 1 ):
            aa_i_mask = (aa_assignments == aa_i)  # pick out the active areas

            if num_dims == 1:
                mean_aai_signals = np.mean( proc_signals[ aa_i_mask ], axis=0, keepdims=True )
                proc_signals_Tcomp[ aa_i_mask ] -= mean_aai_signals  # T compensation

            elif num_dims == 2:
                mean_aai_signals = np.mean( proc_signals[ :, aa_i_mask ], axis=1, keepdims=True )
                proc_signals_Tcomp[ :, aa_i_mask ] -= mean_aai_signals  # T compensation

        # for

        return proc_signals_Tcomp

    # temperature_compensation

# class: FBGNeedle


def __get_argparser() -> argparse.ArgumentParser:
    """ Parser for cli arguments"""
    # Setup parsed arguments
    parser = argparse.ArgumentParser( description="Make a new/Update an existing needle FBG parameter" )

    parser.add_argument( '--update-params', type=str, default=None, help='Update the FBG needle parameter file',
                         dest='update_file' )

    needle_diam_grp = parser.add_mutually_exclusive_group( required=False )
    needle_diam_grp.add_argument( '--needle-gauge', type=str, default=None, help='The gauge of the needle (eg. 18G)' )
    needle_diam_grp.add_argument( '--diameter', type=float, default=None, help='The diameter of the needle (in mm)' )

    material_grp = parser.add_argument_group( title="Material Properties" )
    material_grp.add_argument( '--material', type=str, default=None, help="The material of the needle." )
    material_grp.add_argument( '--Emod', type=float, default=None, help="Young's Modulus of the needle (in GPa)" )
    material_grp.add_argument( '--poisson-ratio', type=float, default=None, help="The Poisson's ratio of the needle" )

    parser.add_argument( 'length', type=float, help='The entire length of the FBG needle' )
    parser.add_argument( 'num_channels', type=int, help='The number of channels in the FBG needle' )

    parser.add_argument( 'sensor_locations', type=float, nargs='+', help='Sensor locations from the tip of the needle' )
    parser.add_argument( 'needle_num', type=int, help='The number of channels in the FBG needle' )

    return parser


# __get_argparser

def main( args=None ):
    parser = __get_argparser()
    pargs = parser.parse_args( args )

    # FBG needle parameters
    length = pargs.length  # mm
    num_chs = pargs.num_channels
    # aa_locs_tip = np.cumsum( [ 11, 20, 35, 35 ] )[ ::1 ]  # 4 AA needle
    aa_locs = (length - np.array( pargs.sensor_locations )).tolist()

    needle_num = pargs.needle_num
    serial_number = "{:d}CH-{:d}AA-{:04d}".format( num_chs, len( aa_locs ), needle_num )
    directory = os.path.join( 'data', serial_number )

    # process size parameters
    if pargs.needle_gauge is not None:
        diameter = Needle.DIAM_GAUGE[ pargs.needle_gauge ]

    else:
        diameter = pargs.diameter

    # process material properties
    if pargs.material is not None and pargs.Emod is None and pargs.poisson_ratio is None:
        Emod = Needle.EMOD[ pargs.material ]
        pratio = Needle.P_RATIO[ pargs.material ]

    # if
    elif pargs.material is not None and (pargs.Emod is not None or pargs.poisson_ratio is not None):
        raise ValueError( "Only can set material type by itself. Can't do custom material properties and material." )

    # elif
    else:
        Emod = pargs.Emod
        pratio = pargs.poisson_ratio

    # else

    # get FBGNeedle instance
    if pargs.update_file is not None:
        save_file = os.path.normpath( pargs.update_file )
        directory = os.path.dirname( save_file )
        print( "Updated needle parameters:" )
        needle = FBGNeedle.load_json( save_file )

        # update the needle parameters
        needle._length = length
        needle._serial_number = serial_number
        needle._num_channels = num_chs
        needle._sensor_location = aa_locs
        needle.diameter = diameter
        needle.Emod = Emod
        needle.pratio = pratio

    # if

    else:
        save_file = os.path.join( directory, '../needle_params.json' )
        print( "New needle parameters:" )
        needle = FBGNeedle( length, serial_number, num_chs, aa_locs, diameter=diameter,
                            Emod=Emod, pratio=pratio )

    # else

    print( needle )
    print()

    if not os.path.isdir( directory ):
        os.mkdir( directory )

    # if

    if not os.path.isfile( save_file ) or (pargs.update_file is not None):
        needle.save_json( save_file )
        print( f"Saved needle parameter json: {save_file}" )

    # if
    else:
        raise OSError( "Needle parameter file already exists: {}".format( save_file ) )

    # else


# main

# for debugging purposes and creating new FBGneedle param files
if __name__ == "__main__" or False:
    main()

# if: __main__
