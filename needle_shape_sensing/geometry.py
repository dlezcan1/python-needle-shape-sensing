"""

Library of rigid body mechanical geometry functions.

Author: Dimitri Lezcano

"""

from typing import Union

import numpy as np
from spatialmath import SO2, SO3, SE3


def arclength( points: np.ndarray, axis: int = 0 ):
    """ Compute the arclength of a curve

        :param points: An N x M numpy array  of curve points
        :param axis: the axis in which the points are given by

        :return: (total length, arclength for each point

    """
    # if (axis == 1):
    #     points = points.T # transpose the matrix
    #
    # # if

    # compute differences
    dpoints = np.diff( points, axis=axis )
    ds = np.linalg.norm( dpoints, axis=(axis + 1) % 2 )

    # compute lengths
    s = np.append( [ 0 ], np.cumsum( ds ) )

    return np.max( s ), s


# arclength

def axangle2rotm( theta: Union[ float, np.ndarray ], w: np.ndarray ) -> np.ndarray:
    """ Compute a rotation matrix from the axis angle representation

        :param theta: the angle of rotation (in radians)
        :param w: the unit vector of the direction of rotation

        :return: 3x3 Rotation matrix about w for theta rads
    """

    return rodrigues( theta * w )


# axangle2rotm


def hat( x: Union[ int, float, np.ndarray ] ) -> np.ndarray:
    """ Perform the inverse vee operation

        Args:
            x: a float or a numpy 1-d array of size [0, 1,3,6]

        Returns:
            2x2 so(2) element if x.size == 0
            3x3 so(3) element if x.size == 3
            4x4 se(3) element if x.size == 6
    """
    if isinstance( x, (int, float) ):
        X = hat_so2( x )

    else:
        assert (x.ndim == 1)

        if x.size == 1:  # so(2)
            X = hat_so2( x[ 0 ] )

        elif x.size == 3:  # so(3)
            X = hat_so3( x )

        elif x.size == 6:  # se(3)
            X = hat_se3( x )

        else:
            raise IndexError( "Size of `x` must be either 1, 3, or 6 vector" )

    # else

    return X


# hat


def hat_so2( x: float ) -> np.ndarray:
    """ Perform inverse vee operation for so(2)"""

    return np.array( [ [ 0, -x ], [ x, 0 ] ] )


# hat_so2

def hat_so3( x: np.ndarray ) -> np.ndarray:
    """ Perform inverse vee operation for so(3)"""
    assert (x.ndim == 1 and x.size == 3)

    X = np.zeros( (3, 3) )
    X[ 0, 1 ] = -x[ 2 ]
    X[ 0, 2 ] = x[ 1 ]
    X[ 1, 2 ] = -x[ 0 ]

    X -= X.T

    return X


# hat_so3

def hat_se3( x: np.ndarray ) -> np.ndarray:
    """ Perform inverse vee operation for se(3)
        Args:
            x = [angular; linear] = [w;v]

        Return:
            4x4 se(3) element array

    """
    assert (x.ndim == 1 and x.size == 6)

    X = np.zeros( (4, 4) )

    # perform rotational term
    X[ 0:3, 0:3 ] = hat_so3( x[ 0:3 ] )

    # perform linear term
    X[ 0:3, -1 ] = x[ 3:6 ]

    return X


# hat_se3

def is_so2( X: np.ndarray ) -> bool:
    """ Criteria for so2 matrix

        - 2x2 matrix
        - skew-symmetric

    """

    return X.shape == (2, 2) and is_skewsymm( X )


# is_so2

def is_SO2( R: np.ndarray ) -> bool:
    """ Determine if R is an element of SO(3)"""
    try:
        SO2( R, check=True )
        retval = True

    except ValueError:
        retval = False

    return retval


# is_SO2

def is_so3( X: np.ndarray ) -> bool:
    """ Criteria for so2 matrix

        - 3x3 matrix
        - skew-symmetric

    """

    return X.shape == (3, 3) and is_skewsymm( X )


# is_so3

def is_SO3( R: np.ndarray ) -> bool:
    """ Determine if R is an element of SO(3)"""
    try:
        SO3( R, check=True )
        retval = True

    except ValueError:
        retval = False

    return retval


# is_SO3


def is_se3( X: np.ndarray ) -> bool:
    """ Criteria for so2 matrix

        - 4x4 matrix
        - so3 upper left block
        - bottom row of zeros

    """

    return X.shape == (4, 4) and is_skewsymm( X[ 0:3, 0:3 ] ) and np.all( X[ -1, : ] == 0 )


# is_so3

def is_SE3( H: np.ndarray ) -> bool:
    """ Determine if R is an element of SO(3)"""
    try:
        SE3( H, check=True )
        retval = (H.shape == (4, 4))

    except ValueError:
        retval = False

    return retval


# is_SE3


def is_skewsymm( X: np.ndarray ) -> bool:
    """ Determine if a matrix is skew-symmetric"""

    return np.all( X.T == -X )


# is_skewsymm

def is_symm( X: np.ndarray ) -> bool:
    """ Determine if a matrix is symmetric """

    return np.all( X.T == X )


# is_symm

def quat2rotm( q: np.ndarray ) -> np.ndarray:
    """ Convert a unit quaternion to a """
    q = q.astype( float )
    q /= np.linalg.norm( q )  # ensure unit

    # unpack quaternion
    c, v = q[ 0 ], q[ 1:4 ]  # cos(theta/2), sin(theta/2) * w
    s = np.linalg.norm( v )

    # compute angle and vector
    theta = 2 * np.arctan2( s, c )
    if theta == 0:
        R = np.eye( 3 )  # identity matrix

    else:
        w = v / s

        # compute rotation matrix
        R = axangle2rotm( theta, w )

    # else

    return R


# quat2rotm


def rodrigues( w: np.ndarray ) -> np.ndarray:
    """ Compute the rodrigues formula for the vector

        :param w: 3D vector of rotation

        :return: 3x3 Rotation matrix

    """
    theta = np.linalg.norm( w )
    if theta == 0:
        R = np.eye( 3 )

    else:
        wh = hat_so3( w / theta )  # skew(w)
        R = np.eye( 3 ) + np.sin( theta ) * wh + (1 - np.cos( theta )) * (wh @ wh)

    return R


# rodrigues


def rotm2axangle( R: np.ndarray ) -> (float, np.ndarray):
    """ Convert a rotation matrix to a quaternion
        :param R: numpy array of rotation matrix

        :return: angle of rotation, 3-D vector unit vector
    """
    theta = float( np.arccos( (np.trace( R ) - 1) / 2 ) )
    if theta == 0:
        w = np.array( [ 1, 0, 0 ] )

    else:
        w = 1 / (2 * np.sin( theta )) * vee_so3( R - R.T )

    return theta, w.astype( np.float )


# rotm2axangle

def rotm2quat( R: np.ndarray ) -> np.ndarray:
    """ Convert a rotation matrix to a quaternion
        :param R: numpy array of rotation matrix

        :return: 4-D vector unit quaternion of (w, x, y, z)
    """

    theta, w = rotm2axangle( R )
    w = w.astype( float )
    if np.linalg.norm( w ) > 0:
        w /= np.linalg.norm( w )

    return np.append( np.cos( theta / 2 ), np.sin( theta / 2 ) * w )


# rotm2quat

def rot2d( t: float ) -> np.ndarray:
    """ 2D rotation matrix"""

    return np.array( [ [ np.cos( t ), -np.sin( t ) ], [ np.sin( t ), np.cos( t ) ] ] )


# rot2d

def rotx( t: float ) -> np.ndarray:
    """ Rotation matrix about x-axis"""
    return np.array( [ [ 1, 0, 0 ], [ 0, np.cos( t ), -np.sin( t ) ], [ 0, np.sin( t ), np.cos( t ) ] ] )


# rotx

def roty( t: float ) -> np.ndarray:
    """ Rotation matrix about y-axis"""
    return np.array( [ [ np.cos( t ), 0, np.sin( t ) ], [ 0, 1, 0 ], [ -np.sin( t ), 0, np.cos( t ) ] ] )


# roty

def rotz( t: float ) -> np.ndarray:
    """ Rotation matrix about z-axis"""
    return np.array( [ [ np.cos( t ), -np.sin( t ), 0 ], [ np.sin( t ), np.cos( t ), 0 ], [ 0, 0, 1 ] ] )


# rotz

def vee( X: np.ndarray ) -> Union[ float, np.ndarray ]:
    """ Perform vee operation on X

        Args:
            X: numpy array of either so(2), so(3), se(3)

        Return:
            float if X is so(2)
            numpy vector if X is so(3) or se(3)

    """

    if is_so2( X ):
        x = vee_so2( X )

    elif is_so3( X ):
        x = vee_so3( X )

    elif is_se3( X ):
        x = vee_se3( X )

    else:
        raise TypeError( "'X' is not so(2), so(3) or se(3)" )

    return x


# vee

def vee_so2( X: np.ndarray, validate: bool = False ) -> float:
    """ Perform vee operation on X

        Args:
            X: 2x2 so(2) array
            validate: check whether X is so(2) or not

        Return:
            float of vee(X) X[1,0]

    """

    if validate:
        assert (is_so2( X ))

    return X[ 1, 0 ]


# vee_so2

def vee_so3( X: np.ndarray, validate: bool = False ) -> np.ndarray:
    """ Perform vee operation on X

        Args:
            X: 3x3 so(3) array
            validate: boolean on whether to check if X is so(3)

        Return:
            float of vee(X) [ X[ 2, 1 ], X[ 0, 2 ], X[ 1, 0 ] ]

    """

    if validate:
        assert (is_so3( X ))

    x = np.array( [ X[ 2, 1 ], X[ 0, 2 ], X[ 1, 0 ] ] )

    return x


# vee_so3

def vee_se3( X: np.ndarray, validate: bool = False ) -> np.ndarray:
    """ Perform vee operation on X

        Args:
            X: 4x4 se(3) array
            validate: boolean on whether to check if X is so(3)

        Return:
            float of vee(X) [ vee_so3(X[0:3,0:3]), x[0:3,-1]]

    """

    if validate:
        assert (is_se3( X ))

    x = np.append( vee_so3( X[ 0:3, 0:3 ], validate=validate ), X[ 0:3, -1 ] )

    return x

# vee_se3
