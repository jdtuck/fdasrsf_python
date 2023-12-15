"""
Author: Kanad Khanna
2021-11-24
Translation of MATLAB interpolation code to Python.
"""

import numpy as np
import scipy.interpolate
import scipy.integrate


def interparc(t, px, py, *args, method='spline'):
    """
    % usage: pt, f_pt = interparc(t,px,py)    % a 2-d curve
    % usage: pt, f_pt = interparc(t,px,py,pz) % a 3-d curve
    % usage: pt, f_pt = interparc(t,px,py,pz,pw,...) % a 4-d or higher dimensional curve
    % usage: pt, f_pt = interparc(t,px,py,method) % a 2-d curve, method is specified
    %
    % Interpolates new points at any fractional point along
    % the curve defined by a list of points in 2 or more
    % dimensions. The curve may be defined by any sequence
    % of non-replicated points.
    %
    % arguments: (input)
    %  t   - vector of numbers, 0 <= t <= 1, that define
    %        the fractional distance along the curve to
    %        interpolate the curve at. t = 0 will generate
    %        the very first point in the point list, and
    %        t = 1 yields the last point in that list.
    %        Similarly, t = 0.5 will yield the mid-point
    %        on the curve in terms of arc length as the
    %        curve is interpolated by a parametric spline.
    %
    %        If t is a scalar integer, at least 2, then
    %        it specifies the number of equally spaced
    %        points in arclength to be generated along
    %        the curve.
    %
    %  px, py, pz, ... - vectors of length n, defining
    %        points along the curve. n must be at least 2.
    %        Exact Replicate points should not be present
    %        in the curve, although there is no constraint
    %        that the curve has replicate independent
    %        variables.
    %
    %  method - (OPTIONAL) string flag - denotes the method
    %        used to compute the points along the curve. 
    %        may be 'linear' or 'spline'
    %        
    %        method == 'linear' --> Uses a linear chordal
    %               approximation to interpolate the curve.
    %               This method is the most efficient.
    %        method == 'spline' --> Uses a parametric spline
    %               approximation for the interpolation in
    %               arc length. Generally for a smooth curve,
    %               this method may be most accurate.
    %
    %        DEFAULT: 'spline'
    %
    %
    % arguments: (output)
    %  pt - Interpolated points at the specified fractional
    %        distance (in arc length) along the curve.
    %  f_pt - Evaluation of the function at the resampled points
    % Author: John D'Errico
    % e-mail: woodchips@rochester.rr.com
    % Release: 1.0
    % Release date: 3/15/2010
    """     
    # Method may be any of {'linear', 'spline'}
    if method not in ('linear', 'spline'):
        raise ValueError("INTERPARC: Incorrect method. Method must be either 'linear' or 'spline'.")
                
    # If t is an integer, generate t evenly spaced points in [0,1]
    if isinstance(t, int) and t > 1:
        t = np.linspace(0, 1, t)
    # Otherwise, create a column vector from the array that was passed
    else:
        t  = np.array(t).reshape(-1,1)
        
    # Create column vectors from px and py, as well
    px = np.array(px).reshape(-1,1)
    py = np.array(py).reshape(-1,1)
    
    # Make sure that only valid t's were passed
    if ((t < 0).any() or (t > 1).any()):
        raise ValueError("ARCLENGTH: Improper t. All elements of t must be 0 <= t <= 1.")
            
    # How many points will be interpolated?
    nt = len(t)
    
    # The number of points on the curve itself
    n = len(px)
    
    # Are px and py both vectors of the same length?
    if len(px) != len(py):
        raise ValueError("ARCLENGTH: Improper px or py. px and py must be vectors of the same length.")
    if len(px) < 2:
        raise ValueError("ARCLENGTH: Improper px or py. px and py must be vectors of length at least 2.")
    
    # Compose px and py into a single array. This way, if more dimensions are
    # provided, the extension is trivial.
    pxy = np.hstack([px, py])                           # indexing is pxy[timestep, dimension]
    ndim = 2
    
    # Are there any other arguments?
    if len(args) > 0:
        # Create column vectors from pz, pw, etc.
        args = [np.array(p_other).reshape(-1,1) for p_other in args]
        # Stack all dimensions into a single matrix.
        for arg in args:
            pxy = np.column_stack([pxy, arg])
        ndim = pxy.shape[1]
        
    # Pre-allocate the result, pt
    # pt = np.full((nt, ndim), np.NaN)
    
    # Compute the chordal (linear) arclength of each segment. This will be
    # needed for any of the methods.
    chordlen = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
    
    # Normalize the arclengths to a unit total
    chordlen = chordlen / chordlen.sum()
    
    # Cumulative arclength
    cumarc = np.concatenate(([0], np.cumsum(chordlen)))
    
    
    # The linear method (trivial case)
    if method == 'linear':
        # Which interval did each point fall in, in terms of t?
        tbins = np.digitize(t, cumarc)
        
        # Catch any problems at the ends
        tbins[(tbins <= 0) | (t <= 0)] = 1
        tbins[(tbins >= n) | (t >= 1)] = n - 1
        
        # Convert indices to 0-based indexing
        tbins = tbins - 1
        
        # Interpolate
        s = (t - cumarc[tbins]) / chordlen[tbins]
        
        # Be nice, and allow the code to work on older releases that don't have bsxfun
        # NOTE: Actually, numpy does automatic broadcasting :) Just add a dimension to s.
        pt = pxy[tbins, :] + (pxy[(tbins + 1), :] - pxy[tbins, :]) * np.expand_dims(s,1)
            
        # Create the spline as a piecewise linear function
        # spl = []
        # for i in range(ndim):
        #     coefs = [np.diff(pxy[:,i]) / np.diff(cumarc),   pxy[:-1, i]]
        #     coefs = np.vstack(coefs)
        #     spl.append(scipy.interpolate.PPoly(c=coefs, x=cumarc))
            
        # # Evaluate the interpolated points
        # f_pt = np.vstack([f_i(pt) for f_i in spl]).T
        
        return np.squeeze(pt)
        
    
    # Spline method
    elif method == 'spline':
        spl  = []
        for i in range(ndim):
            # Compute parametric splines
            spl.append(scipy.interpolate.CubicSpline(x=cumarc, y=pxy[:,i]))
            nc = spl[i].c.size
            if nc < 4:
                # Just pretend it has cubic segments
                # NOTE: The matlab code just edits a struct. This is an object and
                # may not behave the same, so we probably need to construct a new
                # object.
                raise NotImplementedError('Fewer than 4 coefficients created. Kanad has not implemented this case yet.')
            
        # And now differentiate them
        spld = [spline.derivative() for spline in spl]
        
        # Catch the case where there were exactly 3 points in the curve, and
        # spline was used to generate the interpolant. In this case, spline
        # creates a curve with only 1 piece, not 2.
        if cumarc.size == 3:
            cumarc = spl[0].x
            n = cumarc.size
            chordlen = chordlen.sum()
            
        # Generate the total arclength along the curve by integrating each segment
        # and summing the results. The integration scheme does its job using an
        # ODE solver.
        seglen = np.zeros((n-1, 1))
        for i in range(n - 1):
            # Extract polynomials for the derivatives.
            # NOTE: scipy.CubicSpline stores its coefs in reverse (descending) order
            polyarray = [np.polynomial.polynomial.Polynomial(coef=spld[j].c[:, i][::-1]) for j in range(ndim)]
                
            # Integrate the arclength for the i'th segment using ode45 for the integral.
            # I could have done this part with quad too, but then it would not have been
            # perfectly (numerically) consistent with the next operation in this tool.
            results = scipy.integrate.solve_ivp(
                lambda t, y: np.sqrt(np.sum(np.square([P(t) for P in polyarray]),
                                            axis=0)),           # sqrt((dx/dt)^2 + (dy/dt)^2 + ...)
                t_span=(0, chordlen[i]),
                y0=[0,],
                rtol=1.e-9
            )
            seglen[i] = results.y[:,-1]
            
        # and normalize the segments to have unit total length
        totalsplinelength = seglen.sum()
        cumseglen = np.concatenate(([0], seglen.cumsum()))
        
        # Which interval did each point fall into, in terms of t, but relative to the
        # cumulative arc lengths along the parametric spline?
        tbins = np.digitize(t * totalsplinelength, cumseglen)
        
        # Catch any problems at the ends
        tbins[(tbins <= 0) | (t <= 0)] = 1
        tbins[(tbins >= n) | (t >= 1)] = n - 1
        
        # Convert indices to 0-based indexing
        tbins = tbins - 1
        
        # Do the fractional integration within each segment for the interpolated
        # points. t is the parameter used to define the splines. It is defined in 
        # terms of a linear chordal arclength. This works nicely when a linear 
        # piecewise interpolant was used. However, what is asked for is an arclength 
        # interpolation in terms of arclength of the spline itself. Call s the
        # arclength traveled along the spline.
        s = totalsplinelength * t
        
        # The ode45 solver will now include an events callback so we can catch zero crossings
        ode_events = lambda t, y: y
        ode_events.terminal = True
        ode_events.direction = 1
        
        ti = t.copy()
        for i in range(nt):
            # si is the piece of arc length that we will look for in this spline segment
            si = np.squeeze(s[i] - cumseglen[tbins[i]])
            
            # Extract polynomials for the derivatives in the interval the point lies in
            # NOTE: scipy.CubicSpline stores its coefs in reverse (descending) order
            polyarray = [np.polynomial.polynomial.Polynomial(coef=np.squeeze(spld[j].c[:, tbins[i]][::-1])) for j in range(ndim)]
            
            # We need to integrate in t, until the integral crosses the specified value of 
            # si. Because we have defined totalsplinelength, the lengths will be normalized 
            # at this point to a unit length.
            #
            # Start the ode solver at -si, so we will just look for an event where y crosses 0.
            results = scipy.integrate.solve_ivp(
                lambda t, y: np.sqrt(np.sum(np.square([P(t) for P in polyarray]),
                                            axis=0)),           # sqrt((dx/dt)^2 + (dy/dt)^2 + ...)
                t_span=(0, chordlen[tbins[i]]),
                y0=[-si,],
                rtol=1.e-9,
                events=ode_events
            )
            tout = results.t
            yout = results.y.flatten()  # Will this be a problem? Assuming k==1 always.
            te   = results.t_events[0]
            
            # We only need that point where a zero crossing occurred.
            # If no crossing was found, then we can look at each end.
            if len(te) > 0:
                ti[i] = te[0] + cumarc[tbins[i]]
            else:
                # A crossing must have happened at the very beginning or the 
                # end, and the ode solver missed it, not trapping that event.
                if np.abs(yout[0]) < np.abs(yout[-1]):
                    # The event must have been at the start.
                    ti[i] = tout[0] + cumarc[tbins[i]]
                else:
                    # The event must have been at the end.
                    ti[i] = tout[-1] + cumarc[tbins[i]]
                    
        # Interpolate the parametric splines at ti to get our interpolated value.
        pt = np.hstack([f(ti) for f in spl])
        
        return pt
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example 1:
    # Interpolate a set of unequally spaced points around the perimeter of
    # a unit circle, generating equally spaced points around the perimeter
    # theta = np.sort(np.random.rand(15) * 2 * np.pi)
    # theta = np.append(theta, theta[0])
    theta = np.array([0.49196421, 1.30404303, 1.35277569, 1.66329111, 1.87935206,
       2.83274271, 3.39728439, 3.50759802, 3.87032075, 4.43891395,
       4.55715715, 5.04428418, 5.18313692, 5.59810929, 5.85648442,
       0.49196421])
    px = np.cos(theta)
    py = np.sin(theta)
    
    # Interpolate using parametric splines
    pt, _ = interparc(100, px, py, 'spline')
    
    # Plot the result
    fig, ax = plt.subplots()
    ax.plot(px, py, 'r*')
    ax.plot(pt[:,0], pt[:,1], 'b-o')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Points in blue are uniform in arclength around the circle')
    plt.show()
