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

from . import fbg_signal_processing, numerical, intrinsics, geometry


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
                - serial_number: the Seria number of the needle
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

    def __init__( self, length: float, serial_number: str, num_channels: int, sensor_location: list = [ ],
                  calibration_mats: dict = { }, weights: dict = { }, **kwargs ):
        """
        Constructor

        Args:
            - length: float, of the length of the entire needle (mm)
            - num_channels: int, the number of channels there are
            - sensor_location: list, the arclength locations (mm) of the AA's (default = None)
                This measurement is from the base of the needle
        """

        # data checking
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

    def aa_idx( self, aax: str ):
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

    def assignments_AA( self ) -> list:
        """ Instance method of assignments_aa """
        return FBGNeedle.assignments_aa( self.num_channels, self.num_activeAreas )

    # assignments_AA

    @staticmethod
    def assignments_ch( num_channels: int, num_active_areas: int ) -> list:
        """ Function for returning a list of CH assignments"""
        return sum( [ num_active_areas * [ ch_i ] for ch_i in range( 1, num_channels + 1 ) ], [ ] )

    # assignments_ch

    def assignments_CH( self ) -> list:
        """ Instance method of assignments_ch """
        return FBGNeedle.assignments_ch( self.num_channels, self.num_activeAreas )

    # assignments_ch

    def calculate_length_measured_instance( self, L: float, tip: bool = True, valid: bool = False ):
        """ Determine (and return) which lengths are valid for the current FBGNeedle

            :param s_m: the measured arclengths
            :param L:   the insertion depth
            :param tip: (Default = True) whether the measured arclengths are from the tip of the needle or not.
            :param needle_length: (Default = None) float of the entire needle length. Only needed if tip = False
            :param valid: (Default = False) whether to only return valid arclengths
        """
        # nump-ify the sensor locations
        if tip:
            s_m = np.array( self.sensor_location_tip )
        else:
            s_m = np.array( self.sensor_location )

        return FBGNeedle.calculate_length_measured( s_m, L, tip=tip, needle_length=self.length, valid=valid )

    # calculate_length_measured_instance

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
        curvatures = { }
        aa_assignments = self.assignments_AA()

        if isinstance( raw_signals, dict ):
            proc_signals = np.zeros( self.num_channels * self.num_activeAreas )
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
                proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                                                                               self.num_activeAreas )

            # if

            curvatures = self.curvatures_processed( proc_signals )

        # if

        elif isinstance( raw_signals, np.ndarray ):
            # process the signals
            proc_signals = fbg_signal_processing.process_signals( raw_signals, self.ref_wavelengths )
            if temp_comp:
                proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                                                                               self.num_activeAreas )

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
                mask_aa_i = list( map( lambda aa: aa == aa_i, self.assignments_AA() ) )

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

    def generate_chaa( self ) -> (list, list, list):
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


# class: FBGNeedle

class ShapeSensingFBGNeedle( FBGNeedle ):
    def __init__( self, length: float, serial_number: str, num_channels: int, sensor_location: list = [ ],
                  calibration_mats: dict = { }, weights: dict = { }, ds: float = 0.5, current_depth: float = 0,
                  **kwargs ):
        super().__init__( length, serial_number, num_channels, sensor_location=sensor_location,
                          calibration_mats=calibration_mats, weights=weights, **kwargs )

        # current insertion parameters
        self.current_depth = current_depth
        self.__current_shapetype = intrinsics.SHAPETYPES.SINGLEBEND_SINGLELAYER
        self.current_wavelengths = -np.ones_like( self.ref_wavelengths )
        self.current_curvatures = -np.ones( (len( sensor_location ), 2) )
        self.insertion_parameters = { }

        # define needle shape-sensing optimizers
        self.optimizer = numerical.NeedleParamOptimizations( self, ds=ds )

    # __init__

    @property
    def current_shapetype( self ):
        return self.__current_shapetype

    # property: current_shapetype

    @property
    def ds( self ):
        return self.optimizer.ds

    # property: ds

    @ds.setter
    def ds( self, ds ):
        self.optimizer.ds = ds

    # property setter: ds

    @staticmethod
    def from_FBGNeedle( fbgneedle: FBGNeedle, **kwargs ):
        """ Turn an FBGNeedle into a shape-sensing FBGNeedle

            :param fbgneedle: FBGNeedle to turn into a sensorized one
            :keyword ds: the ds for the ShapeSensingFBGNeedle constructor
            :keyword current_depth: the current insertion depth for the ShapeSensingFBGNeedle constructor

            :return: ShapeSensingFBGNeedle with the current FBGNeedle
        """
        return ShapeSensingFBGNeedle( fbgneedle.length, fbgneedle.serial_number, fbgneedle.num_channels,
                                      sensor_location=fbgneedle.sensor_location,
                                      calibration_mats=fbgneedle.cal_matrices,
                                      weights=fbgneedle.weights, **kwargs )

    # from_FBGNeedle

    def get_needle_shape( self, *args, **kwargs ):
        """ Determine the 3D needle shape of the current shape-sensing needle

        Example (Single-Bend Single-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, w_init_i, R_init=R_init)

        Example (Single-Bend Double-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, kc2_i, w_init_i, R_init=R_init)

        Example (Double-Bend Single-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, w_init_i, R_init=R_init)

        :param kcx_i: the initial kappa_c value(s)
        :param w_init_i: (Default = None) the omega_init value
        :keyword R_init: (Default = numpy.eye(3)) the initial angular offset
        :returns: (N x 3 position matrix of the needle shape, N x 3 x 3 orientation matrices of the needle shape)
                  (None, None) if:
                        - sensors are not calibrated
                        - current insertion depth not > 0
                        - current shape type is not an implemented shape type
        """
        # kwargs get
        R_init = kwargs.get( 'R_init', np.eye( 3 ) )
        s = np.arange( 0, self.current_depth + self.ds, self.ds )

        # initial checks
        pmat, Rmat = None, None
        if not self.sensor_calibrated or np.any( self.current_wavelengths < 0 ):  # check current wavelengths
            pass

        elif self.current_depth <= 0:  # check insertion depth
            pass

        elif self.current_shapetype in intrinsics.SHAPETYPES:
            # initalization
            k0, k0prime, w_init = None, None, None

            if self.current_shapetype == intrinsics.SHAPETYPES.CONSTANT_CURVATURE:
                # determine parameters
                curvature = self.optimizer.constant_curvature( self.current_curvatures, self.current_depth )

                pmat, Rmat = intrinsics.ConstantCurvature.shape( s, curvature )

                pmat = pmat @ R_init.T
                Rmat = R_init @ Rmat

            # if
            elif self.current_shapetype == intrinsics.SHAPETYPES.SINGLEBEND_SINGLELAYER:
                # get parameters
                kc_i = args[ 0 ]
                if len( args ) > 1:
                    w_init_i = args[ 1 ]
                else:
                    w_init_i = np.array( [ kc_i, 0, 0 ] )

                # else

                # determine parameters
                kc, w_init, _ = self.optimizer.singlebend_singlelayer_k0( kc_i, w_init_i, self.current_curvatures,
                                                                          self.current_depth, R_init=R_init, **kwargs )
                # determine k0 and k0prime
                k0, k0prime = intrinsics.SingleBend.k0_1layer( s, kc, self.current_depth, )

            # if: single-bend single-layer

            elif self.current_shapetype == intrinsics.SHAPETYPES.SINGLEBEND_DOUBLELAYER:
                # get parameters
                z_crit = self.insertion_parameters[ 'z_crit' ]
                kc1_i = args[ 0 ]
                kc2_i = args[ 1 ]
                if len( args ) > 2:
                    w_init_i = args[ 2 ]
                else:
                    w_init_i = np.array( [ kc1_i, 0, 0 ] )

                # else

                # determine parameters
                kc1, kc2, w_init, _ = self.optimizer.singlebend_doublelayer_k0( kc1_i, kc2_i, w_init_i,
                                                                                self.current_curvatures,
                                                                                self.current_depth, z_crit=z_crit,
                                                                                R_init=R_init )
                s_crit = intrinsics.SingleBend.determine_2layer_boundary( kc1, self.current_depth, z_crit, self.B,
                                                                          w_init=w_init, s0=0, ds=self.ds,
                                                                          R_init=R_init )

                # determine k0 and k0prime
                k0, k0prime = intrinsics.SingleBend.k0_2layer( s, kc1, kc2, self.current_depth, s_crit )
                Rz = geometry.rotz( np.pi )
                R_init = R_init @ Rz  # rotate the needle 180 degrees about its axis

            # elif: single-bend double-layer

            elif self.current_shapetype == intrinsics.SHAPETYPES.DOUBLEBEND_SINGLELAYER:
                # get parameters
                s_crit = self.insertion_parameters[ 's_double_bend' ]
                kc_i = args[ 0 ]
                if len( args ) > 1:
                    w_init_i = args[ 1 ]
                else:
                    w_init_i = np.array( [ kc_i, 0, 0 ] )

                # else

                kc, w_init, _ = self.optimizer.doublebend_singlelayer_k0( kc_i, w_init_i, self.current_curvatures,
                                                                          self.current_depth, s_crit, R_init=R_init )
                k0, k0prime = intrinsics.DoubleBend.k0_1layer( s, kc, self.current_depth, s_crit=s_crit )

            # elif: double-bend single-layer

            else:
                k0, k0prime, w_init = None, None, None

            # else: Cannot find paramterization

            # pmat and Rmat
            if (k0 is not None) and (k0prime is not None) and (w_init is not None):
                # compute w0 and w0prime
                w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
                w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

                # integrate
                pmat, Rmat, _ = numerical.integrateEP_w0( w_init, w0, w0prime, self.B, s0=0, ds=self.ds, R_init=R_init,
                                                          arg_check=False )

            # if

        # elif: shape-sensing

        return pmat, Rmat

    # get_needle_shape

    @staticmethod  # overloaded
    def load_json( filename: str ):
        """ Load a ShapeSensingFBGNeedle from a needle parameter json file"""
        return ShapeSensingFBGNeedle.from_FBGNeedle( super().load_json( filename ) )

    # load_json

    def update_shapetype( self, shapetype, *args ):
        """ Update the current needle shape-sensing type

            Example:
                ss_fbgneedle.update_shapetype(intrinsics.SHAPETYPES.SINGLEBEND_SINGLELAYER)

            Example:
                ss_fbgneedle.update_shapetype(intrinsics.SHAPETYPES.SINGLEBEND_DOUBLELAYER) -> throws IndexOutOfBoundsError
                ss_fbgneedle.update_shapetype(intrinsics.SHAPETYPES.SINGLEBEND_DOUBLELAYER, z_crit) -> OK

            :param shapetype: listing from current implemented shape-types
                if shapetype is SINGLEBEND_DOUBLELAYER:
                    first argument must be length of the first layer

                if shapetype is DOUBLEBENG_SINGLELAYER:
                    first argument must the double bend insertion depth


            :returns: True if update was successful, False otherwise.

        """
        if shapetype == intrinsics.SHAPETYPES.CONSTANT_CURVATURE:
            self.__current_shapetype = shapetype
            success = True

        # if
        elif shapetype == intrinsics.SHAPETYPES.SINGLEBEND_SINGLELAYER:
            self.__current_shapetype = shapetype
            success = True

        # elif
        elif shapetype == intrinsics.SHAPETYPES.SINGLEBEND_DOUBLELAYER:
            self.__current_shapetype = shapetype
            self.insertion_parameters[ 'z_crit' ] = args[ 0 ]
            success = True

        # elif
        elif shapetype == intrinsics.SHAPETYPES.DOUBLEBEND_SINGLELAYER:
            self.__current_shapetype = shapetype
            self.insertion_parameters[ 's_double_bend' ] = args[ 0 ]
            success = True

        # elif
        else:
            success = False

        # else

        return success

    # update_shapetype

    def update_wavelengths( self, wavelengths: np.ndarray, reference: bool = False, temp_comp: bool = True ):
        """ Update the current signals and curvatures with the updated value. This will also determine the
            curvatures if the current reference signals are set

            :param wavelengths: numpy array to update the current signals with
            :param reference: (Default = False) whether to update the reference wavelengths
            :param temp_comp: (Default = True) whether to perform temperature compensation on the signals

        """
        # get mean value if it is an array of measurements
        if wavelengths.ndim == 2:
            wavelengths = np.mean( wavelengths, axis=0 )

        # if

        # check for wavlength shape to mach reference wavelengths shape
        if wavelengths.shape != self.ref_wavelengths.shape:
            raise IndexError( f"Signals must be of size {self.ref_wavelengths.shape}." )

        # if

        # set the wavelengths
        if reference:
            self.ref_wavelengths = wavelengths
            self.current_wavelengths = wavelengths

        # if
        else:
            self.current_wavelengths = wavelengths

        # else

        # calculate the curvatures
        if self.sensor_calibrated:
            curvatures = self.curvatures_raw( self.current_wavelengths, temp_comp=temp_comp )
            self.current_curvatures = curvatures

        # if
        else:
            curvatures = None

        # else

        return wavelengths, curvatures  # return the signals & curvatures anyways

    # update_wavelengths


# class: ShapeSensingFBGNeedle


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
        save_file = os.path.join( directory, 'needle_params.json' )
        print( "New needle parameters:" )
        Needle( 1, 2, diameter=1, Emod=1, pratio=1 )
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
