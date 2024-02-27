import warnings

from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np
import scipy.integrate.odepack
import scipy.optimize as spoptim

from needle_shape_sensing import cost_functions, intrinsics
from needle_shape_sensing.sensorized_needles import FBGNeedle

class NeedleParamOptimizations:
    warnings.warn("NeedleParamOptimizations is legacy. Move to NeedleShapeOptimizer Class", DeprecationWarning, stacklevel=2)
    def __init__(
            self, fbgneedle: FBGNeedle, ds: float = 0.5, optim_options: dict = None,
            continuous: bool = True ):
        assert (ds > 0)
        self.fbg_needle = fbgneedle

        # don't calculate each time
        self.__needle_B = self.fbg_needle.B
        self.__needle_Binv = np.linalg.inv( self.fbg_needle.B )

        self.ds = ds

        # optimizer options
        default_options = {
            'w_init_bounds': [ (-0.01, 0.01) ] * 3,
            'kc_bounds'    : [ (0, 0.01) ],
            'tol'          : 1e-8,
            'method'       : 'SLSQP',
            }
        self.options = default_options
        self.options.update( optim_options if optim_options is not None else { } )

        # integration options
        self.continuous = continuous

    # __init__

    def constant_curvature( self, data: np.ndarray, L: float ):
        """
            Compute the constant curvature

        """
        assert (L > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = np.ones_like( s_data_ins )

        # else

        # normalize the weights
        weights = weights / np.sum( weights )

        # compute the weighted-mean curvature from the measurements
        data_ins_w = data_ins * weights.reshape( -1, 1 )
        curvature = data_ins_w.sum( axis=0 )

        return curvature

    # constant_curvature

    def singlebend_singlelayer_k0(
            self, k_c_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            R_init: np.ndarray = np.eye( 3 ), **kwargs ):
        """
            :param k_c_0: initial intrinsic curvature
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc, w_init optimized, optimization results
        """
        # argument checking
        assert (L > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keysShape(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ k_c_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.singlebend_singlelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, self.__needle_B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.singlebend_singlelayer_cost(
            eta=eta,
            data=data_ins,
            s_m=s_data_ins,
            ds=self.ds,
            B=self.fbg_needle.B,
            scalef=c_f,
            arg_check=False,
            continuous=self.continuous,
            **cost_kwargs
        )
        res = self.__optimize( cost_fn, eta_0, **kwargs )
        kc, w_init = res.x[ 0 ], res.x[ 1:4 ]

        return kc, w_init, res

    # singlebend_singlelayer_k0

    def singlebend_doublelayer_k0(
            self, kc1_0: float, kc2_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            s_crit: float = None, z_crit: float = None, R_init: np.ndarray = np.eye( 3 ),
            **kwargs ):
        """
            :param kc1_0: initial intrinsic curvature for layer 1
            :param kc2_0: initial intrinsic curvature for layer 2
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param z_crit: (Default = None) the length of the first layer
            :param s_crit: (Default = None) the length of the needle in first layer
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc1, kc2, w_init optimized, optimization results
        """
        assert (L >= z_crit > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ kc1_0, kc2_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                's_crit' : s_crit,
                'z_crit' : z_crit,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.singlebend_doublelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, self.__needle_B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.singlebend_doublelayer_cost(
                eta, data_ins, s_data_ins, self.ds,
                self.fbg_needle.B,
                scalef=c_f, arg_check=False,
                continuous=self.continuous,
                **cost_kwargs )
        res = self.__optimize( cost_fn, eta_0, **kwargs )
        kc1, kc2, w_init = res.x[ 0 ], res.x[ 1 ], res.x[ 2:5 ]

        return kc1, kc2, w_init, res

    # singlebend_doublelayer_k0

    def doublebend_singlelayer_k0(
            self, kc_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            s_crit: float, R_init: np.ndarray = np.eye( 3 ), **kwargs ):
        """
            :param kc_0: initial intrinsic curvature for layer 1
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param s_crit: (Default = None) the length of the needle in first layer
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc1, kc2, w_init optimized, optimization results
        """
        assert (L >= s_crit > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ kc_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.doublebend_singlelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, s_crit,
                self.fbg_needle.B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.doublebend_singlelayer_cost(
                eta, data_ins, s_data_ins, self.ds,
                s_crit, self.__needle_B,
                scalef=c_f, arg_check=False,
                continuous=self.continuous,
                **cost_kwargs )
        res = self.__optimize( cost_fn, eta_0, **kwargs )

        kc, w_init = res.x[ 0 ], res.x[ 1:4 ]

        return kc, w_init, res

    # doublebend_singlelayer_k0

    def __optimize( self, cost_fn, eta_0, **kwargs ):
        """ Optimize wrapper for multiple functions"""
        optim_options = self.options.copy()
        optim_options.update( kwargs )

        # filter specific bounds and add bounds if not already specified
        exclude_keys = [ 'w_init_bounds', 'kc_bounds', 'needle_rotations' ]
        optim_options = dict(
            filter(
                lambda x: x[ 0 ] not in exclude_keys,
                optim_options.items()
            )
        )
        if optim_options.get( 'bounds' ) is None:
            bounds = (
                self.options[ 'kc_bounds' ] * (eta_0.size - 3) 
                + self.options[ 'w_init_bounds' ]
            )
            optim_options[ 'bounds' ] = bounds

        # bounds

        # perform optimization
        with warnings.catch_warnings():
            warnings.simplefilter( "ignore", scipy.integrate.odepack.ODEintWarning )
            warnings.simplefilter( "ignore", RuntimeWarning )
            result = spoptim.minimize( cost_fn, eta_0, **optim_options )

        # with

        return result

    # __optimize


# NeedleParamOptimizations
    
class NeedleShapeOptimizer:
    def __init__(
        self, 
        fbgneedle: FBGNeedle, 
        ds: float = 0.5, 
        optim_options: dict = None,
        continuous: bool = True 
    ):
        assert (ds > 0), "ds must be > 0!"
        self.fbg_needle = fbgneedle

        # don't calculate each time
        self.__needle_B    = self.fbg_needle.B
        self.__needle_Binv = np.linalg.inv( self.fbg_needle.B )

        self.ds = ds

        # optimizer options
        default_options = {
            'w_init_bounds': [ (-0.01, 0.01) ] * 3,
            'kc_bounds'    : [ (0, 0.01) ],
            'tol'          : 1e-8,
            'method'       : 'SLSQP',
            }
        self.options = default_options
        self.options.update( optim_options if optim_options is not None else { } )

        # integration options
        self.continuous = continuous

    # __init__
        
    def optimize_curvature_measurements(
        self,
        curvature_measurements: List[Tuple[float, np.ndarray]],
        shape_type: intrinsics.SHAPETYPE,
        shape_parameters: intrinsics.ShapeParametersBase,
        initial_estimate: intrinsics.ShapeParametersBase = None,
    ):
        """
        
        Args:
            curvature_measurements: List of (measurement points)
        
        """
        pass # TODO

    # optimize_curvature_measurements

    def __optimize_costfn( self, cost_fn, eta_0, **kwargs ):
        """ Optimize wrapper for multiple functions"""
        optim_options = self.options.copy()
        optim_options.update( kwargs )

        # filter specific bounds and add bounds if not already specified
        exclude_keys = [ 'w_init_bounds', 'kc_bounds', 'needle_rotations' ]
        optim_options = dict(
            filter(
                lambda x: x[ 0 ] not in exclude_keys,
                optim_options.items()
            )
        )
        if optim_options.get( 'bounds' ) is None:
            bounds = (
                self.options[ 'kc_bounds' ] * (eta_0.size - 3) 
                + self.options[ 'w_init_bounds' ]
            )
            optim_options[ 'bounds' ] = bounds

        # bounds

        # perform optimization
        with warnings.catch_warnings():
            warnings.simplefilter( "ignore", scipy.integrate.odepack.ODEintWarning )
            warnings.simplefilter( "ignore", RuntimeWarning )
            result = spoptim.minimize( cost_fn, eta_0, **optim_options )

        # with

        return result

    # __optimize

# class: NeedleShapeOptimizer