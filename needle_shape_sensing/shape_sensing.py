import numpy as np

from needle_shape_sensing import numerical, intrinsics, geometry, sensorized_needles


class ShapeSensingFBGNeedle( sensorized_needles.FBGNeedle ):
    def __init__(
            self, length: float, serial_number: str, num_channels: int, sensor_location=None,
            calibration_mats=None, weights=None, ds: float = 0.5, current_depth: float = 0,
            optim_options: dict = None, cts_integration: bool = False, **kwargs ):
        super().__init__(
                length,
                serial_number,
                num_channels,
                sensor_location=sensor_location,
                calibration_mats=calibration_mats,
                weights=weights,
                **kwargs
        )

        # current insertion parameters
        self.current_depth        = current_depth
        self.__current_shapetype  = intrinsics.SHAPETYPE.SINGLEBEND_SINGLELAYER
        self.current_wavelengths  = -np.ones_like( self.ref_wavelengths )
        self.current_curvatures   = np.zeros( (2, self.num_activeAreas), dtype=self.ref_wavelengths.dtype )
        self.insertion_point      = np.zeros((3,), dtype=self.ref_wavelengths.dtype) # relative to needle
        self.insertion_parameters = dict()

        # define needle shape-sensing optimizers
        self.optimizer = numerical.NeedleParamOptimizations(
            self,
            ds=ds,
            optim_options=optim_options,
            continuous=cts_integration
        )
        self.current_kc = [ 0 ]
        self.current_rotations = None # [ 0 ] * int(self.length // self.ds + 1)  # radians
        self.current_winit = np.zeros( 3 )

    # __init__

    def __repr__( self ):
        return "Shape Sensing " + super().__repr__()

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

    @property
    def continuous_integration( self ):
        return self.optimizer.continuous

    # continuous_integration

    @continuous_integration.setter
    def continuous_integration( self, continuous: bool ):
        self.optimizer.continuous = continuous

    # property setter: continuous_integration

    @staticmethod
    def from_FBGNeedle( fbgneedle: sensorized_needles.FBGNeedle, **kwargs ):
        """ Turn an FBGNeedle into a shape-sensing FBGNeedle

            :param fbgneedle: FBGNeedle to turn into a sensorized one
            :keyword ds: the ds for the ShapeSensingFBGNeedle constructor
            :keyword current_depth: the current insertion depth for the ShapeSensingFBGNeedle constructor

            :return: ShapeSensingFBGNeedle with the current FBGNeedle
        """
        if isinstance(fbgneedle, sensorized_needles.MCFNeedle):
            return ShapeSensingMCFNeedle.from_MCFNeedle(fbgneedle, **kwargs)

        return ShapeSensingFBGNeedle(
                fbgneedle.length,
                fbgneedle.serial_number,
                fbgneedle.num_channels,
                sensor_location=fbgneedle.sensor_location,
                calibration_mats=fbgneedle.cal_matrices,
                weights=fbgneedle.weights,
                diameter=fbgneedle.diameter,
                Emod=fbgneedle.Emod,
                pratio=fbgneedle.pratio,
                ref_wavelengths=fbgneedle.ref_wavelengths,
                **kwargs
        )

    # from_FBGNeedle

    def get_needle_shape( self, *args, **kwargs ):
        """ Determine the 3D needle shape of the current shape-sensing needle within a specific insertion depth

        Example (Single-Bend Single-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, w_init_i, R_init=R_init)

        Example (Single-Bend Double-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, kc2_i, w_init_i, R_init=R_init)

        Example (Double-Bend Single-Layer)
            pmat, Rmat = ss_fbgneedle.get_needle_shape(kc_i, w_init_i, R_init=R_init)

        :param kcx_i: the initial kappa_c value(s)
        :param w_init_i: (Default = None) the omega_init value
        :keyword R_init: (Default = numpy.eye(3)) the initial angular offset
        :keyword air_cubic_fit: (Default = False) whether to handle air-gap with
        :keyword return_wv: (Default = False) whether to return wv or not
        :returns: (N x 3 position matrix of the needle shape, N x 3 x 3 orientation matrices of the needle shape, N x 3 curvature matrix (if return_wv=True))
                  (None, None, None (if return_wv=True)) if:
                        - sensors are not calibrated
                        - current insertion depth not > 0
                        - current shape type is not an implemented shape type
        """
        # kwargs get
        R_init = kwargs.get( 'R_init', np.eye( 3 ) )
        s = np.arange( 0, self.current_depth + self.ds, self.ds )
        current_rotations = None if self.current_rotations is None else self.current_rotations[ -int(self.current_depth // self.ds - 1): ]


        # initial checks
        pmat, Rmat, wv = None, None, None
        if not self.sensor_calibrated:  # check current wavelengths
            return pmat, Rmat

        if self.current_shapetype not in intrinsics.SHAPETYPE:
            return pmat, Rmat

        # handle air-gap
        pmat_air, Rmat_air, wv_air = None, None, None
        if self.insertion_point[2] > self.ds:
            s_air = np.arange(
                0,
                (self.insertion_point[2]//self.ds + 1) * self.ds,
                self.ds
            )
            pmat_air, Rmat_air = intrinsics.AirDeflection.shape(
                s_air,
                self.insertion_point,
                cubic_fit=kwargs.get("air_cubic_fit", False),
            )
            # Rmat_air = np.tile(np.eye(3), (pmat_air.shape[0], 1, 1)) # FIXME: update with model-based method
            wv_air   = np.zeros_like(pmat_air) # FIXME: update with model-based method

        # if

        if self.current_depth <= 0:  # check insertion depth
            return pmat_air, Rmat_air

        # initalization
        k0, k0prime, w_init = None, None, None

        if self.current_shapetype == intrinsics.SHAPETYPE.CONSTANT_CURVATURE:
            # determine parameters
            curvature = self.optimizer.constant_curvature(
                    self.current_curvatures.T, self.current_depth )
            curvature = np.append( curvature, 0 )  # ensure 3 vector
            pmat, Rmat = intrinsics.ConstantCurvature.shape( s, curvature )

            pmat = pmat @ R_init.T
            Rmat = R_init @ Rmat
            wv   = np.tile(curvature, (pmat.shape[0], 1))

            self.current_winit = curvature
            self.current_kc    = [np.linalg.norm(curvature)]

        # if
        elif self.current_shapetype == intrinsics.SHAPETYPE.SINGLEBEND_SINGLELAYER:
            # get parameters
            kc_i = args[ 0 ]
            if len( args ) > 1:
                w_init_i = args[ 1 ]
                if not isinstance( w_init_i, np.ndarray ) or len( w_init_i ) != 3:
                    raise ValueError( "w_init_i must be a 3-D vector" )
            else:
                w_init_i = np.array( [ kc_i, 0, 0 ] )

            # else

            # determine parameters
            kc, w_init, _ = self.optimizer.singlebend_singlelayer_k0(
                kc_i,
                w_init_i,
                self.current_curvatures.T,
                self.current_depth,
                R_init=R_init,
                needle_rotations=current_rotations,
                **kwargs
            )
            self.current_kc = [ kc ]
            self.current_winit = w_init

            # determine k0 and k0prime
            k0, k0prime = intrinsics.SingleBend.k0_1layer(
                s,
                kc,
                self.current_depth,
                return_callable=self.continuous_integration
            )

        # if: single-bend single-layer

        elif self.current_shapetype == intrinsics.SHAPETYPE.SINGLEBEND_DOUBLELAYER:
            # get parameters
            z_crit = self.insertion_parameters[ 'z_crit' ]
            kc1_i = args[ 0 ]
            kc2_i = args[ 1 ]
            if len( args ) > 2:
                w_init_i = args[ 2 ]
                if not isinstance( w_init_i, np.ndarray ) or len( w_init_i ) != 3:
                    raise ValueError( "w_init_i must be a 3-D vector" )
            else:
                w_init_i = np.array( [ kc1_i, 0, 0 ] )

            # else

            # determine parameters
            kc1, kc2, w_init, _ = self.optimizer.singlebend_doublelayer_k0(
                kc1_i,
                kc2_i,
                w_init_i,
                self.current_curvatures.T,
                self.current_depth,
                z_crit=z_crit,
                R_init=R_init,
                needle_rotations=current_rotations,
            )
            self.current_kc = [ kc1, kc2 ]
            self.current_winit = w_init
            s_crit = intrinsics.SingleBend.determine_2layer_boundary(
                kc1,
                self.current_depth,
                z_crit,
                self.B,
                w_init=w_init,
                s0=0,
                ds=self.ds,
                R_init=R_init,
                needle_rotations=current_rotations,
                continuous=self.continuous_integration,
            )

            # determine k0 and k0prime
            k0, k0prime = intrinsics.SingleBend.k0_2layer(
                s,
                kc1,
                kc2,
                self.current_depth,
                s_crit,
                return_callable=self.continuous_integration,
            )
            Rz     = geometry.rotz( np.pi )
            R_init = R_init @ Rz  # rotate the needle 180 degrees about its axis

        # elif: single-bend double-layer

        elif self.current_shapetype == intrinsics.SHAPETYPE.DOUBLEBEND_SINGLELAYER:
            # get parameters
            s_crit = self.insertion_parameters[ 's_double_bend' ]
            kc_i = args[ 0 ]
            if len( args ) > 1:
                w_init_i = args[ 1 ]
                if not isinstance( w_init_i, np.ndarray ) or len( w_init_i ) != 3:
                    raise ValueError( "w_init_i must be a 3-D vector" )
            else:
                w_init_i = np.array( [ kc_i, 0, 0 ] )

            # else

            kc, w_init, _ = self.optimizer.doublebend_singlelayer_k0(
                kc_i,
                w_init_i,
                self.current_curvatures.T,
                self.current_depth,
                s_crit,
                R_init=R_init,
            )
            self.current_kc = [ kc ]
            self.current_winit = w_init
            k0, k0prime = intrinsics.DoubleBend.k0_1layer(
                s,
                kc,
                self.current_depth,
                s_crit=s_crit,
                return_callable=self.continuous_integration,
            )

        # elif: double-bend single-layer

        else:
            return pmat, Rmat

        # else: Cannot find parameterization


        # compute w0, w0prime and shape
        if self.continuous_integration:
            w0      = lambda s: np.append( k0( s ), [ 0, 0 ] )
            w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

            pmat, Rmat, wv = numerical.integrateEP_w0_ode(
                w_init,
                w0,
                w0prime,
                self.B,
                s0=0,
                ds=self.ds,
                needle_rotations=self.current_rotations,
                R_init=R_init,
                arg_check=False,
            )
        # if
        else:
            w0      = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
            w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

            pmat, Rmat, wv = numerical.integrateEP_w0(
                w_init,
                w0,
                w0prime,
                self.B,
                s0=0,
                ds=self.ds,
                needle_rotations=self.current_rotations,
                R_init=R_init,
                arg_check=False,
            )
        # if

        # stack the air layer
        if (pmat_air is not None) and (Rmat_air is not None) and (wv_air is not None):
            pmat = pmat @ Rmat_air[-1].T  + pmat_air[np.newaxis, -1]
            Rmat = Rmat_air[np.newaxis, -1] @ Rmat

            pmat = np.concatenate((pmat_air, pmat[1:]), axis=0)
            Rmat = np.concatenate((Rmat_air, Rmat[1:]), axis=0)
            wv   = np.concatenate((wv_air,   wv[1:]),   axis=0)

        # if

        elif self.insertion_point[2] > 0:
            pmat += self.insertion_point[np.newaxis]

        # elif

        return pmat, Rmat

    # get_needle_shape

    @staticmethod  # overloaded
    def load_json( filename: str ):
        """ Load a ShapeSensingFBGNeedle from a needle parameter json file"""
        return ShapeSensingFBGNeedle.from_FBGNeedle(
                super( ShapeSensingFBGNeedle, ShapeSensingFBGNeedle ).load_json( filename ) )

    # load_json

    def set_rotation( self, L: float, rot_rads: float ):
        """ Set the needle rotation at the specified length

            :param L: float of needle depth when the needle the rotation was performed
            :param rot_rads: float of the amount of rotation/orientation of the needle

            :returns: boolean of whether operation was successful or not

        """
        # check for valid length
        if (L < 0) or L > self.length:
            return False

        # find closest length
        lengths = np.arange( 0, self.length + self.ds, self.ds )
        L_closest_idx = np.argmin( np.abs( lengths - L ) ).item()

        # update the rotation # TODO: is this correct?
        # for i in range(L_closest_idx, len(self.current_rotations)): # update after points
        for i in range(L_closest_idx): # update before points
            self.current_rotations[i] = rot_rads

        return True

    # set_rotation

    def update_curvatures( self, processed: bool = False, temp_comp: bool = True ):
        """ Update the current curvatures

            :param processed: (Default = False) a boolean on whether the current signals are processed or not
            :param temp_comp: (Default = True) a boolean on whether to perform temperature compensated for non-processed signals

            :returns: True if update was successful, False otherwise.

        """
        if self.sensor_calibrated and self.is_calibrated:
            success = True

            if processed:
                self.current_curvatures = 1e-3 * self.curvatures_processed( self.current_wavelengths )

            else:
                self.current_curvatures = 1e-3 * self.curvatures_raw(
                        self.current_wavelengths, temp_comp=temp_comp )

        # if
        else:
            success = False

        # else

        return success

    # update_curvatures

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
        if shapetype == intrinsics.SHAPETYPE.CONSTANT_CURVATURE:
            self.__current_shapetype = shapetype
            self.insertion_parameters = { }
            success = True

        # if
        elif shapetype == intrinsics.SHAPETYPE.SINGLEBEND_SINGLELAYER:
            self.__current_shapetype = shapetype
            self.insertion_parameters = { }
            success = True

        # elif
        elif shapetype == intrinsics.SHAPETYPE.SINGLEBEND_DOUBLELAYER:
            self.__current_shapetype = shapetype
            self.insertion_parameters = { 'z_crit': args[ 0 ] }
            success = True

        # elif
        elif shapetype == intrinsics.SHAPETYPE.DOUBLEBEND_SINGLELAYER:
            self.__current_shapetype = shapetype
            self.insertion_parameters = { 's_double_bend': args[ 0 ] }
            success = True

        # elif
        else:
            success = False

        # else

        return success

    # update_shapetype

    def update_wavelengths(
            self, wavelengths: np.ndarray, reference: bool = False, temp_comp: bool = True,
            processed: bool = False ):
        """ Update the current signals and curvatures with the updated value. This will also determine the
            curvatures if the current reference signals are set

            :param wavelengths: numpy array to update the current signals with
            :param reference: (Default = False) whether to update the reference wavelengths
            :param temp_comp: (Default = True) whether to perform temperature compensation on the signals
            :param processed: (Default = False) whether the signals are processed or not

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
        success    = self.update_curvatures( processed=processed, temp_comp=temp_comp )
        curvatures = self.current_curvatures if success else None

        return wavelengths, curvatures  # return the signals & curvatures anyways

    # update_wavelengths

# class: ShapeSensingFBGNeedle

class ShapeSensingMCFNeedle(ShapeSensingFBGNeedle, sensorized_needles.MCFNeedle):
    def __init__(
        self,
        length:
        float,
        serial_number: str,
        num_channels: int,
        sensor_location=None,
        calibration_mats=None,
        central_core_ch:int = None,
        weights=None,
        ds: float = 0.5,
        current_depth: float = 0,
        optim_options: dict = None,
        cts_integration: bool = False,
        **kwargs
    ):
        super().__init__(
            length=length,
            serial_number=serial_number,
            num_channels=num_channels,
            sensor_location=sensor_location,
            calibration_mats=calibration_mats,
            weights=weights,
            ds=ds,
            current_depth=current_depth,
            optim_options=optim_options,
            cts_integration=cts_integration,
            central_core_ch=central_core_ch,
            **kwargs
        )

    # __init__

    @staticmethod
    def from_FBGNeedle( fbgneedle: sensorized_needles.FBGNeedle, **kwargs ):
        raise NotImplementedError("use 'from_MCFNeedle' function")

    # from_FBGNeedle

    @staticmethod
    def from_MCFNeedle( mcfneedle: sensorized_needles.MCFNeedle, **kwargs ):
        """ Turn an FBGNeedle into a shape-sensing FBGNeedle

            :param mcfneedle: MCFNeedle to turn into a sensorized one
            :keyword ds: the ds for the ShapeSensingMCFNeedle constructor
            :keyword current_depth: the current insertion depth for the ShapeSensingMCFNeedle constructor

            :return: ShapeSensingMCFNeedle with the current MCFNeedle
        """
        return ShapeSensingMCFNeedle(
                mcfneedle.length,
                mcfneedle.serial_number,
                mcfneedle.num_channels,
                sensor_location=mcfneedle.sensor_location,
                calibration_mats=mcfneedle.cal_matrices,
                weights=mcfneedle.weights,
                diameter=mcfneedle.diameter,
                Emod=mcfneedle.Emod,
                pratio=mcfneedle.pratio,
                ref_wavelengths=mcfneedle.ref_wavelengths,
                central_core_ch=mcfneedle.central_core_ch,
                **kwargs
        )

    # from_MCFNeedle

# ShapeSensingMCFNeedle