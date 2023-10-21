import numpy as np
import scipy.signal as signal
import scipy.optimize as opt
import matplotlib.pyplot as plt
from TSA.analysis import plotACFnPACF, mat2np
from TSA.tests import whiteness_test
from filterpy.kalman import KalmanFilter
from statsmodels.tools.numdiff import approx_hess1, approx_hess2, approx_hess3
from nfoursid.nfoursid import NFourSID

class PEM:
    """
    Parameter Estimation for general ARMAX model using Prediction Error Minimization.

    Model: A(z)y(t) = [B(z)/F(z)]x(t) + [C(z)/D(z)]e(t)

    Attributes:
    - y: Endogenous signal.
    - x: Exogenous signal (default=None for ARMA models).
    - A, B, C, D, F: Initial polynomial guesses or polynomial orders (guesses default to 0 in that case).
    - *_guess: Polynomials based on input parameters.
    - *_free: Boolean arrays indicating which polynomial coefficients are adjustable.
    - *_est: Estimated polynomial coefficients after fitting.
    - nA, nB, nC, nD, nF: Sizes of respective polynomials.
    - nFree: Number of adjustable parameters.
    - theta_est: Vector representation of model coefficients.

    Methods:
    - fit: Estimate parameters by minimizing prediction error.
        Returns instance of PEMResult with details about the estimation.
    - rpem: Recursive Parameter Estimation using a Kalman Filter approach.
    - set_free_params: Set which polynomial coefficients are adjustable.
    """
    def __init__(self, y, x=None, A=[1], B=[], C=[1], D=[1], F=[1]):
        self.y = y

        # If given as polynomial or if given as model order
        self.A_guess = np.concatenate(([1],np.zeros(A))) if isinstance(A,int) else np.array(A)
        self.B_guess = np.concatenate(([1],np.zeros(B))) if isinstance(B,int) else np.array(B)
        self.C_guess = np.concatenate(([1],np.zeros(C))) if isinstance(C,int) else np.array(C)
        self.D_guess = np.concatenate(([1],np.zeros(D))) if isinstance(D,int) else np.array(D)
        self.F_guess = np.concatenate(([1],np.zeros(F))) if isinstance(F,int) else np.array(F)

        # Ensure A and C start with 1
        assert self.A_guess[0] == 1, "The leading coefficient of the A polynomial must be 1."
        assert self.C_guess[0] == 1, "The leading coefficient of the C polynomial must be 1."

        if x is None:
            assert len(self.B_guess) == 0, "B should not be provided if x is not given"
            assert len(self.F_guess) == 1, "F should not be provided if x is not given"
        else:
            assert len(self.B_guess) > 0, "B should be provided if x is given"


        assert self.D_guess[0] == 1, "The leading coefficient of the D polynomial must be 1."
        assert self.F_guess[0] == 1, "The leading coefficient of the F polynomial must be 1."
        
        self.isX = False if x is None else True
        self.x = np.zeros(len(y)) if x is None else x

        assert len(self.y)==len(self.x), "x and y should have the same length."
        
        self.nA = len(self.A_guess) # Sizes of polynomials
        self.nB = len(self.B_guess)
        self.nC = len(self.C_guess)
        self.nD = len(self.D_guess)
        self.nF = len(self.F_guess)

        self.A_free = np.concatenate(([0],np.ones(self.nA-1))).astype(bool) # Set free params (init is all free except first)
        self.B_free = np.ones(self.nB).astype(bool) # Default is all free for B
        self.C_free = np.concatenate(([0],np.ones(self.nC-1))).astype(bool)
        self.D_free = np.concatenate(([0],np.ones(self.nD-1))).astype(bool)
        self.F_free = np.concatenate(([0],np.ones(self.nF-1))).astype(bool)

        self.nFree = sum([sum(f) for f in [self.A_free,self.B_free,self.C_free,self.D_free,self.F_free]])
        # assert self.nFree > 0, "Model has to estimate at least one parameter"

        self.A_est = np.copy(self.A_guess) # Estimated after fitting, init is the same as guess
        self.B_est = np.copy(self.B_guess)
        self.C_est = np.copy(self.C_guess)
        self.D_est = np.copy(self.D_guess)
        self.F_est = np.copy(self.F_guess)

        self.theta_est = self._polys_to_theta(self.A_est, self.B_est, self.C_est, self.D_est, self.F_est)

    def fit(self, method='LS', bh=False, bh_iter=100):
        """
        Estimates model parameters by minimizing the prediction error. The method utilizes either 
        local optimization methods such as least squares or basin-hopping for global optimization.

        Arguments:
        - method (str): The optimization method to use.
            Default is 'LS' for least squares minimization.
            Methods from scipy.optimize.minimize can also be used, 'L-BFGS-B' is recommended for speed.
        - bh (bool): Whether to use the basin-hopping global optimization method.
            If False, the method looks for a local minimum using the specified method.
            Default is False.
        - bh_iter (int): Number of basin-hopping iterations if bh is True. Default is 100.

        Returns:
        - PEMResult: An object containing:
            * theta (np.ndarray): Estimated model parameters in vector form.
            * polys (dict): Dictionary containing estimated polynomial coefficients (A, B, C, D, F).
            * polys_free (dict): Dictionary indicating which polynomial coefficients are adjustable.
            * poly_std_errs (dict): Standard errors for each polynomial coefficient.
            * conf_ints (np.ndarray): Confidence intervals for the model parameters.
            * resid (np.ndarray): Residuals from the model fitting.
            * result (object): Full result object from the optimization process.
            * scores (dict): Goodness-of-fit scores (MSE, AIC, BIC, FPE, R^2).
            * model_instance: A reference to the model instance.

        Notes:
        - The 'LS' method indicates a least squares minimization, using 'scipy.optimize.least_squares'.
        - Other methods are directly passed to scipy's minimize function and should be valid minimizers.
        """
        # Concatenate params into theta vector of free params
        theta_init = self._polys_to_theta(self.A_guess, self.B_guess, self.C_guess, self.D_guess, self.F_guess)

        # Minimize sum of squares of prediction error
        if method=='LS':
            if bh:
                result = self._LS_basin_hopping(theta_init, niter=bh_iter)
            else:
                result = opt.least_squares(lambda theta: self._pred_err(theta), theta_init)
        else: # Other methods used in scipy.optimize.minimize
            if bh:
                result = opt.basinhopping(lambda theta: self._sum_of_squares(theta), theta_init, niter=bh_iter, minimizer_kwargs={'method': method}, disp=False, seed=0)
            else:
                result = opt.minimize(lambda theta: self._sum_of_squares(theta), theta_init, method=method)

        self.theta_est = result['x'] # Extract theta

        A_params, B_params, C_params, D_params, F_params = self._theta_to_polys(self.theta_est)
        self.A_est = np.where(self.A_free, A_params, self.A_guess)
        self.B_est = np.where(self.B_free, B_params, self.B_guess)
        self.C_est = np.where(self.C_free, C_params, self.C_guess)
        self.D_est = np.where(self.D_free, D_params, self.D_guess)
        self.F_est = np.where(self.F_free, F_params, self.F_guess)

        std_errs = self._calc_SE(self.theta_est)
        poly_std_errs = {'ABCDF'[i]: self._theta_to_polys(std_errs,fill='zeros')[i] for i in range(5)} # SE in polynomial form

        # Confidence intervals for theta
        conf_ints = self._conf_ints(std_errs)
        # Residuals
        resid = self._pred_err(self.theta_est)

        # Calculate MSE,AIC,BIC,FPE:
        sigma2 = np.var(resid) # Variance of residuals
        n = len(resid)
        logL = self._log_likelihood(self.theta_est) # Log likelihood
        k = self.nFree  # Number of free parameters
        MSE = np.sum(resid**2) / n
        AIC = -2*logL + 2*k
        BIC = -2*logL + k*np.log(n)
        FPE = ((n + k) / (n - k)) * sigma2

        # Calculate NRMSE (normalized rms error) goodness-of-fit
        RMSE = np.sqrt(np.mean(resid**2))
        NRMSE = RMSE/np.std(self.y)
        NRMSE_P = (1-NRMSE)*100 # As percentage

        scores = {
            'FPE': FPE,
            'MSE': MSE,
            'AIC': AIC,
            'BIC': BIC,
            'NRMSE': NRMSE_P
            }
        
        # Package polynomials into dictionary
        polys = {'A': self.A_est,
                    'B': self.B_est,
                    'C': self.C_est,
                    'D': self.D_est,
                    'F': self.F_est}
        
        polys_free = {'A': self.A_free,
                        'B': self.B_free,
                        'C': self.C_free,
                        'D': self.D_free,
                        'F': self.F_free}

        return PEM.PEMResult(self.theta_est, polys, polys_free, std_errs, poly_std_errs, conf_ints, resid, result, scores, self)


    def rpem(self, P=None, Q=None, R=None, k_pred=None):
        """
        Recursive Parameter Estimation for ARMAX models using a Kalman Filter approach.

            Arguments:
            - P (np.ndarray, float, int, optional): Initial error covariance (default based on 'fit' residuals). 
            Also known as 'Rxx_1'.
            - Q (np.ndarray, float, int, optional): Process noise covariance for the Kalman Filter (default is a small value). 
            Also known as 'Re'.
            - R (np.ndarray, float, int, optional): Measurement noise covariance (default based on 'fit' residuals). 
            Also known as 'Rw'.
            - k_pred (int, list of int, optional): Steps ahead for prediction (default is no prediction).
            Supports multiple prediction steps when list of ints is entered.

        The function initializes parameters with the 'fit' method, then recursively estimates and predicts 
        using new data points.

        Returns:
            A dictionary containing 'x' (state estimates), 'ehat' (prediction errors), 'R' (parameter variances)
            and 'ypred' (predictions for k steps ahead).

        Note: Models with nontrivial D or F polynomials are not supported.
        """
        
        assert self.nD<=1 and self.nF<=1, 'Models with D and F polynomials are not yet supported with rpem.'

        init_model = self.fit()
        theta_init = init_model.params

        kf = KalmanFilter(dim_x=self.nFree, dim_z=1)

        y = self.y
        u = self.x

        kf.x = theta_init.reshape(-1,1)  # Initial state
        kf.F = np.eye(self.nFree)  # State transition matrix

        # Are the covariances correctly assumed?
        if Q is None: kf.Q = np.array([[1e-6,0],[0,1e-6]]) # State noise covariance 
        elif isinstance(Q, (float, int)): kf.Q = Q*np.eye(self.nFree)
        else: kf.Q = Q

        kf.R = R if R is not None else np.var(init_model.resid)  # Measurement noise covariance

        if P is None: kf.P = np.diag(init_model.std_errs ** 2) # Initial covariance
        elif isinstance(P, (float, int)): kf.P = P*np.eye(self.nFree)
        else: kf.P = P


        N = len(self.y)
        start = np.max([self.nA,self.nB,self.nC,self.nD,self.nF])-1
        
        Xsave = np.zeros((N, self.nFree, 1))
        Xsave[0,:] = kf.x
        ehat = np.zeros(N)

        k_preds = np.atleast_1d(k_pred) if k_pred is not None else []
        end = np.max(k_preds) if len(k_preds) else 0
        predictions = {k:np.zeros(N) for k in k_preds}
        variances = np.zeros((N, self.nFree,1))

        # Create H matrix easier, a.k.a Ct matrix
        def createH(t):
            Ha = np.array([-y[t-n] for n in np.where(self.A_free)[0]])
            Hb = np.array([u[t-n] for n in np.where(self.B_free)[0]])
            Hc = np.array([ehat[t-n] for n in np.where(self.C_free)[0]])
            H = np.concatenate((Ha,Hb,Hc)).reshape(1,-1)
            return H

        # Kalman loop
        for t in range(start, N-end):
            kf.H = createH(t)
            kf.predict()
            kf.update(y[t])

            ehat[t] = kf.y.ravel()

            # Predictions
            if len(k_preds):
                for k in k_preds:
                    H_pred = createH(t+k)
                    predictions[k][t+k] = np.ravel(H_pred @ kf.x)
            
            variances[t] = np.diag(kf.P).reshape(-1,1)
            Xsave[t] = kf.x

        return {'x': Xsave, 'ehat': ehat, 'vars': variances, 'ypred': predictions}



    def set_free_params(self, A_free=None, B_free=None, C_free=None, D_free=None, F_free=None):
        """
        Sets which polynomial coefficients are adjustable during model fitting.

        Parameters:
        - A_free, B_free, C_free, D_free, F_free (list or None): Lists of booleans that specify 
            which coefficients of the respective polynomials are adjustable during optimization. 
            Each list must match the length of the respective `_guess` attribute. If None, 
            the previous value is retained.

        Notes:
        - The first coefficient for polynomials A, C, D, and F is always set to be non-adjustable.
        - The *_free polynomials can be set with 0 and 1 integers instead of booleans.
        """
        if A_free is None: A_free = self.A_free
        if B_free is None: B_free = self.B_free
        if C_free is None: C_free = self.C_free
        if D_free is None: D_free = self.D_free
        if F_free is None: F_free = self.F_free
        
        if len(A_free) != len(self.A_guess): raise ValueError(f"A_free should have length {len(self.A_guess)}")
        if len(B_free) != len(self.B_guess): raise ValueError(f"B_free should have length {len(self.B_guess)}")
        if len(C_free) != len(self.C_guess): raise ValueError(f"C_free should have length {len(self.C_guess)}")
        if len(D_free) != len(self.D_guess): raise ValueError(f"D_free should have length {len(self.D_guess)}")
        if len(F_free) != len(self.F_guess): raise ValueError(f"F_free should have length {len(self.F_guess)}")

        self.A_free = np.array(A_free).astype(bool)
        self.B_free = np.array(B_free).astype(bool)
        self.C_free = np.array(C_free).astype(bool)
        self.D_free = np.array(D_free).astype(bool)
        self.F_free = np.array(F_free).astype(bool)

        self.A_free[0] = False # Make sure the first param is 1 for ACDF
        self.C_free[0] = False
        self.D_free[0] = False
        self.F_free[0] = False

        self.nFree = np.sum([np.sum(f) for f in [self.A_free,self.B_free,self.C_free,self.D_free,self.F_free]])
        self.theta_est = self._polys_to_theta(self.A_est, self.B_est, self.C_est, self.D_est, self.F_est)

    def _samps_to_remove(self):
        return self.nA+self.nD+self.nF-3

    def _pred_err(self, theta, y=None, x=None):
        A,B,C,D,F = self._theta_to_polys(theta)

        if len(B) == 0:
            B = np.array([0])
        
        if y is None:
            y = self.y

        if x is None:
            x = self.x

        # A*y = [B/F]*x + [C/D]*e
        # => e = [A*D/C]*y - [B*D/F*C]*x
        AxD = np.convolve(A,D)
        BxD = np.convolve(B,D)
        FxC = np.convolve(F,C)
        e = filter(AxD,C,y) - filter(BxD,FxC,x)

        # rmv = self._samps_to_remove()
        # e = e[rmv:]
        return e
            

    def _sum_of_squares(self, theta):
        e = self._pred_err(theta)
        return np.sum(e**2)


    def _log_likelihood(self, theta):
        e = self._pred_err(theta)
        # Variance of residuals
        sigma2 = np.var(e, ddof=len(self.theta_est))
        N = len(e)
        EtE = e.T @ e
        # Log likelihood function
        # logL = -0.5 * n * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * np.sum(e**2)
        logL = -N/2 * np.log(EtE/N) - N/2*1*np.log(2*np.pi) - N/2
        return logL


    def _calc_SE(self, theta):
        """
        Calculate standard errors of theta.

        Current problem: Incorrect standard errors when there is only a B and/or F polynomial estimated.
        (so if any other polynomial besides these is included everything is fine).
        In matlab when this is the case it goes into a function called 'localTSModel' that calculates a
        state space model of the prediction error using n4sid, converts it into a transfer function, then filters
        part of the varaiable 'psi' with it. However, I can not replicate the behaviour of n4sid in python.
        """
        # --Method 1--
        # # Calculate inverse hessian matrix
        # hess = approx_hess1(theta, lambda theta: self._log_likelihood(theta))
        # inv_hess = np.linalg.inv(-hess)
        # # Calculate standard errors
        # SE = np.sqrt(np.diag(inv_hess))


        # --Method 2-- Current method, a bit more consistent than method 1
        J = opt.approx_fprime(theta, lambda theta: self._pred_err(theta)) # Jacobian
        Jt = np.matrix.transpose(J)
        cov = np.linalg.inv(Jt@J)
        # cov = [[1.2104e-04]]
        var_e = (np.var(self._pred_err(theta)))
        SE = np.sqrt(var_e*np.diag(cov))

        # --Method 3-- matlab method
        # r1 = self._matlab_cov_m(theta, inputpoly=False)['J']
        # r2 = self._matlab_cov_m(theta, inputpoly=True)['J']
        # r3 = np.linalg.qr(np.linalg.pinv(r2.T).dot(r1.T))[1]
        # J = r3.dot(r1)
        # Jt = np.matrix.transpose(J)
        # cov = np.linalg.inv(Jt@J)
        # SE = np.sqrt(np.diag(cov))

        return SE

    def _matlab_cov_m(self, theta, inputpoly):
        """
        Attempt to translate the function gntrad in getErrorAndJacobian.m.
        If someone asks me to explain this tell my family i love them
        """
        def n4sid_simple(y, n): 
            """
            WRONG. This function has to replicate the behaviour of n4sid in matlab, returning the
            correct A,C,K matrices and prediction error e.
            """
            # Step 1: Data Matrix Construction
            L = (y.shape[1] - 1) // 2
            Yf = np.vstack([y[:, i:i+L] for i in range(L)])
            Yp = np.vstack([y[:, i:i+L] for i in range(1, L+1)])
            
            # Step 2: Singular Value Decomposition (SVD)
            U, S, Vh = np.linalg.svd(Yf, full_matrices=False)
            U = U[:, :n]
            S = np.diag(S[:n])
            V = Vh.T[:, :n]
            
            # Observability matrix
            Ob = U @ np.sqrt(S)
            
            # Step 3: System Matrix Estimation
            A = np.linalg.pinv(Ob[:-1, :]) @ Ob[1:, :]
            C = Ob[0, :].reshape(1, -1)
            
            # Estimate K using A, C and the data
            X = np.linalg.pinv(Ob) @ Yf
            X_next = A @ X[:, :-1] + C.T @ y[:, 1:L]
            X_cur = X[:, 1:]
            e = X_next - X_cur
            K = e @ np.linalg.pinv(y[:, L:L+1] - C @ X_cur)
            
            return A, C, K, e
        

        A,B,C,D,F = self._theta_to_polys(theta)
        size = len(self.y)
        nparams = len(theta)
        nA = np.sum(self.A_free)
        nB = np.sum(self.B_free)
        nC = np.sum(self.C_free)
        nD = np.sum(self.D_free)
        nF = np.sum(self.F_free)
        nk = np.nonzero(B)[0][0]

        v = filter(A,1,self.y)
        w = filter(B,F,self.x)
        v = v-w
        e = filter(D,C,v)

        if nA: yf = filter(-D,C,self.y)
        if nC: ef = filter(1,C,e)
        if nD: vf = filter(-1,C,v)

        gg = np.convolve(C,F)
        uf = filter(D,gg,self.x)
        wf = filter(-D,gg,w)

        # Compute the gradient PSI
        jj = np.arange(nparams-1,size)
        psi = np.zeros((len(jj), nparams))
        for a in range(nA):
            psi[:,a] = yf[jj-a-1]

        ss = nA
        ss1 = nA+nB+nC+nD
        for b in range(nB):
            I = jj > (b + nk - 1)
            psi[I,ss+b] = uf[jj[I] - b - nk]

        for f in range(nF):
            psi[:,ss1+f] = wf[jj-f-1]

        ss = ss+nB
        ss1 = ss1+nF

        for c in range(nC):
            psi[:,ss+c] = ef[jj-c-1]
        ss = ss+nC
        for d in range(nD):
            psi[:,ss+c] = vf[jj-d-1]

        if inputpoly: # Only goes into this if there is either a B or F polynomial or both
            a,c,k,res = n4sid_simple(e.reshape(1,-1),5)
            
            L = np.linalg.cholesky(np.atleast_2d(np.var(res)))
            num, den = signal.ss2tf(a,k*L,c,L)
            psi = filter(num.ravel(),den.ravel(), psi[::-1, :], axis=0)

        R1 = np.triu(np.linalg.qr(np.hstack([psi, e[jj][:, np.newaxis]]), mode='complete')[1])
        nRr, nRc = R1.shape
        R1 = R1[:min(nRr, nRc), :]

        # R = R1[:nparams, :nparams]
        # Re = R1[:nparams, nparams]

        R = R1[0:min(nparams+1, R1.shape[0]), 0:nparams]
        Re = R1[0:min(nparams+1, R1.shape[0]), nparams][:, np.newaxis] # note the sign difference

        J = R[0:nparams,:]

        return {'EtE': e**2, 'Re': Re, 'R': R, 'J': J}
    

    def _conf_ints(self, errs):
        # Calculate confidence intervals
        CI_lower = self.theta_est - 1.96 * errs
        CI_upper = self.theta_est + 1.96 * errs
        
        CI_pairs = [[l, u] for l, u in zip(CI_lower, CI_upper)]
        return CI_pairs
    
    def _polys_to_theta(self, A, B, C, D, F):
        A_used = A[self.A_free]
        B_used = B[self.B_free]
        C_used = C[self.C_free]
        D_used = D[self.D_free]
        F_used = F[self.F_free]
        return np.concatenate((A_used, B_used, C_used, D_used, F_used))

    def _theta_to_polys(self, theta, fill='guess'):
        # Calculate lengths based on free parameters
        nA_used = np.sum(self.A_free)
        nB_used = np.sum(self.B_free)
        nC_used = np.sum(self.C_free)
        nD_used = np.sum(self.D_free)
        # nF_used = np.sum(self.F_free)

        # Extract values from theta
        A_used = theta[:nA_used]
        B_used = theta[nA_used:nA_used+nB_used]
        C_used = theta[nA_used+nB_used:nA_used+nB_used+nC_used]
        D_used = theta[nA_used+nB_used+nC_used:nA_used+nB_used+nC_used+nD_used]
        F_used = theta[nA_used+nB_used+nC_used+nD_used:]

        if fill == 'guess':
            # Initialize full arrays with guess
            A_full = np.copy(self.A_guess).astype('float64') # Can't converge if i don't set the type???
            B_full = np.copy(self.B_guess).astype('float64')
            C_full = np.copy(self.C_guess).astype('float64')
            D_full = np.copy(self.D_guess).astype('float64')
            F_full = np.copy(self.F_guess).astype('float64')
        elif fill == 'zeros':
            # Initialize full arrays with zeros
            A_full = np.zeros(self.nA)
            B_full = np.zeros(self.nB)
            C_full = np.zeros(self.nC)
            D_full = np.zeros(self.nD)
            F_full = np.zeros(self.nF)            

        # Insert the extracted values into the full arrays
        A_full[self.A_free] = A_used
        B_full[self.B_free] = B_used
        C_full[self.C_free] = C_used
        D_full[self.D_free] = D_used
        F_full[self.F_free] = F_used

        return A_full, B_full, C_full, D_full, F_full

    def _LS_basin_hopping(self, theta_init, niter=100, step_size=0.03):
        """Method doesn't work that well"""
        best_theta = theta_init.copy()
        best_cost = np.inf
        
        current_theta = theta_init

        for _ in range(niter):
            # Local minimization
            result = opt.least_squares(lambda theta: self._pred_err(theta), current_theta)
            current_cost = result['cost']
            
            # Update best_theta and best_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_theta = result['x']

            # Perturb current_theta for next iteration
            current_theta = result['x'] + step_size * np.random.randn(*theta_init.shape)

        return {'x': best_theta, 'cost': best_cost}


    class PEMResult:
        """
        Encapsulates the result after fitting a model via PEM prediction error minimization.

        Attributes:
        - polys (dict): Dictionary containing the estimated polynomial coefficients.
        - A, B, C, D, F (array-like): Specific polynomial coefficients extracted from `polys`.
        - params (array-like): All estimated parameters in a combined theta array.
        - poly_std_errs (dict): Standard errors for each polynomial.
        - conf_ints (dict): Confidence intervals for each polynomial coefficient.
        - polys_free (dict): Boolean masks indicating which coefficients were free during estimation.
        - resid (array-like): Residuals of the model.
        - optimize_res (object): Result object from the optimization process.
        - scores (dict): Dictionary containing various model evaluation metrics.
        - MSE, AIC, BIC, FPE, NRMSE: Specific scores extracted from `scores` for convenience.
        - model (PEM): the instance of the model that was estimated

        Methods:
        - summary: Generates and either prints or returns a summary of the PEM result.
        - predict: Computes a k-step prediction of the data.
        - forecast: Computes a forecast of the n next future values.
        - plot: Plots the negative log-likelihood function against the model parameters.
            For one parameter, a 2D plot is generated, and for two parameters, a 3D surface plot is produced.
        """
        def __init__(self, theta, polys, polys_free, std_errs, poly_std_errs, conf_ints, resid, optimize_res, scores, model):
            self.polys = polys
            self.A = polys['A']
            self.B = polys['B']
            self.C = polys['C']
            self.D = polys['D']
            self.F = polys['F']
            self.params = theta # All estimated parameters in theta array
            self.std_errs = std_errs # Standard errors
            self.poly_std_errs = poly_std_errs
            self.conf_ints = conf_ints # Confidence intervals
            self.polys_free = polys_free # Free parameters
            self.resid = resid # Residual of model
            self.optimize_res = optimize_res
            self.scores = scores
            self.MSE = scores['MSE']
            self.AIC = scores['AIC']
            self.BIC = scores['BIC']
            self.FPE = scores['FPE']
            self.NRMSE = scores['NRMSE']
            self.model = model

        def __str__(self) -> str:
            """Returns a string representation of the PEM result, essentially a summary."""
            return self.summary(return_val=True)


        def predict(self, k=1):
            """
            Computes a k-step prediction of the entire signal.

            Parameters:
            - k (int): Number of steps for prediction. Default is 1.

            Returns:
            - y_hat (array-like): The k-step prediction values.
            """
            x = self.model.x
            y = self.model.y

            yhat = predict_pem(self, y, x, k)
            return yhat
        
        def forecast(self, n=3):
            """
            Computes a forecast of the n next future values.

            Parameters:
            - n (int): Number of future values to forecast. Default is 3.

            Returns:
            - forecasts (array-like): The n future values.
            """
            forecasts = np.zeros(n)
            for t in range(n):
                forecasts[t] = self.predict(k=t+1)[-1]
            
            return forecasts


        def summary(self, return_val=False):
            # Model strings
            models = {
                'A': "AR",
                'C': "MA",
                'AC': "ARMA",
                'AB': "ARX",
                'CB': "MAX",
                'ABC': "ARMAX",
                'ABCDF': "Polynomial"}

            # Active parameters
            active_keys = ''.join(p for p, free in self.polys_free.items() if sum(free))
            if active_keys in models:
                model_name = models[active_keys]
            else:
                model_name = "BJ"

            model_string = f"Discrete-time {model_name} model: "
            equation = []

            if 'A' in active_keys:
                equation.append("A(z)")
            equation.append("y(t) = ")
            equation = [''.join(equation)]

            if 'B' in active_keys:
                if 'F' in active_keys:
                    equation.append("[B(z)/F(z)]x(t)")
                else:
                    equation.append("B(z)x(t)")
            if 'C' in active_keys:
                if 'D' in active_keys:
                    equation.append("[C(z)/D(z)]e(t)")
                else:
                    equation.append("C(z)e(t)")
            if 'D' in active_keys and 'C' not in active_keys:
                equation.append("[1/D(z)]e(t)")
            if 'C' not in active_keys and 'D' not in active_keys:
                equation.append("e(t)")

            model_string += ' + '.join(equation).replace('=  +','=')

            poly_strings = [f'{p}(z) = {self._poly_to_string(self.polys[p],self.poly_std_errs[p])}'  for p in active_keys]

            orders = [f'n{k} = {len(self.polys[k])-1}' for k in active_keys]
            free_coeffs = sum(sum(v) for v in self.polys_free.values())
            order_string = 'Polynomial orders: ' + '    '.join(orders) + '\n' + f'Number of free coefficients: {free_coeffs}'

            est_fit = f'Fit to estimation data (NRMSE): {np.round(self.NRMSE,2)}%'              
            scores = [f'{name} : {np.round(score,3)}' for name,score in self.scores.items()][:-1]
            scores_string = est_fit + '\n' + '  '.join(scores[:2]) + '\n' + '   '.join(scores[2:])

            summary = model_string + '\n'+'\n' + '\n'.join(poly_strings) + '\n'+'\n'+ order_string + '\n' + scores_string +'\n'

            if return_val:
                return summary
            
            print(summary)

        def plotll(self, free_params=None, n_pts=100, lims=None):
            """
            Plots the negative log-likelihood against the model parameters.
            
            Parameters:
            - free_params (list, optional): Parameters to consider as free. Defaults to all.
            - n_pts (int, optional): Number of points for the plot. Defaults to 100.
            - lims (tuple, optional): x and y axis limits. Defaults to (-1.1, 1.1).
            
            Returns:
            None. Displays a 2D or 3D plot based on the number of free parameters.

            Notes:
            A RuntimeWarning might be raised in some cases, indicating the value of the log-
            likelihood is too large. In this case, modify the limits of the plot with 'lims'.
            """
            if free_params is None:
                free_params = np.ones(len(self.params)).astype(bool)
            else:
                free_params = np.array(free_params).astype(bool)
            
            assert np.sum(free_params) in [1,2] , 'Dimensionality too high, set theta_free to have one or two free params.'

            def replace_theta(theta_new_values):
                # Replace free parameters in theta_est with new values
                theta_temp = np.copy(self.params).astype(float)
                theta_temp[free_params] = theta_new_values
                return theta_temp
            
            if np.sum(free_params)==1:
                theta_range = np.linspace(lims[0],lims[1],n_pts)
                thetas = [-self.PEMmodel._log_likelihood(replace_theta(theta)) for theta in theta_range]
                plt.plot(theta_range, thetas)
                plt.xlabel('Theta')
                plt.ylabel('Negative Log-Likelihood')
                plt.title('Minimization function')
                plt.show()
                return

            theta1_range = np.linspace(lims[0], lims[1], n_pts)
            theta2_range = np.linspace(lims[0], lims[1], n_pts)
            Theta1, Theta2 = np.meshgrid(theta1_range, theta2_range)
            ll_values = np.zeros_like(Theta1)
            for i in range(Theta1.shape[0]):
                for j in range(Theta1.shape[1]):
                    theta_new_values = np.array([Theta1[i, j], Theta2[i, j]])
                    ll_values[i, j] = -self.PEMmodel._log_likelihood(replace_theta(theta_new_values))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Theta1, Theta2, ll_values, cmap='viridis')

            ax.set_xlabel('Theta1')
            ax.set_ylabel('Theta2')
            ax.set_zlabel('Negative Log-Likelihood')
            ax.set_title('Minimization function')

            plt.show()
            return

        def _poly_to_string(self, P, errs):
            """Converts polynomial coefficients and their standard errors into a readable string."""
            p = [np.round(float(c), 4) for c in P]
            terms = []

            # Map for converting numbers to their superscript equivalents
            superscript_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
            }

            def to_superscript(num):
                # Convert a number into its superscript representation
                return '⁻' + ''.join([superscript_map[char] for char in str(num)])

            # Constant term
            if p[0] != 0:
                if errs[0] != 0:
                    terms.append(f"{p[0]}(±{np.round(errs[0],4)})")
                else:
                    terms.append(str(p[0]))

            # Remaining terms
            for i, (coef, err) in enumerate(zip(p[1:], errs[1:]), 1):
                if coef == 0:
                    continue
                    
                term_str = ""
                if err != 0:
                    term_str = f'{coef}(±{np.round(err,4)})'
                else:
                    term_str = f'{coef}'
                    
                if coef == 1:
                    term_str = term_str[:-1]  # remove the 1 from the coefficient representation
                elif coef == -1:
                    term_str = term_str[:-1]  # remove the 1 from the coefficient representation, keep minus sign

                term_str += '·z' + to_superscript(i)

                terms.append(term_str)

            return ' + '.join(terms).replace(" + -", " - ")



def filter(B,A,X,remove=False, axis=-1):
    """
    Applies a filter on the form Y = B/A * X.

    Parameters:
    - B (int, list, or ndarray): Numerator coefficients of the filter. If an integer, it will be converted to a list.
    - A (int, list, or ndarray): Denominator coefficients of the filter. If an integer, it will be converted to a list.
    - X (ndarray): Input data to be filtered.
    - remove (bool, int, or str): Determines the removal of initial values in the output.
        * False (default): No values are removed.
        * True: Removes initial values based on the length of B.
        * int: Removes the specified number of initial values.

    Returns:
    - ndarray: Filtered data.
    """
    if isinstance(B, int): B = [B] # If integers are given
    if isinstance(A, int): A = [A]
    if len(B)==0: B = [0]
    if len(A)==0: A = [1]
    B = np.array(B, dtype='float64')
    A = np.array(A, dtype='float64')
    X = np.array(X, dtype='float64')

    Y = signal.lfilter(B,A,X, axis=axis)

    if remove:
        if type(remove)==int: 
            Y = Y[remove:]
        elif type(remove)==bool:
            Y = Y[len(B)-1:]
    return Y


def seasonal(y, season: int, remove=True):
    "Seasonally differentiates the process y to form ∇ₛy where s is the season."
    if season<1: raise ValueError('Season should be at least 1')
    p = np.concatenate(([1], np.zeros(season-1), [-1]))
    return filter(p,1,y, remove=remove)


def estimateBJ(y,x,B=[],d=0,A2=[1],C1=[1],A1=[1], B_free=None, A2_free=None, C1_free=None, A1_free=None,
               titleStr='',noLags='auto', method='LS', bh=False):
    """Estimates the Box-Jenkins model y(t) = [B(z)*z^-d / A2(z)] x(t) + [C1(z)/A1(z)] e(t) using PEM"""

    B_new = np.concatenate((np.zeros(d), B)) if not isinstance(B, int) else np.concatenate((np.zeros(d), np.ones(B)))
    model = PEM(y, x, B=B_new, F=A2, C=C1, D=A1)

    if isinstance(B, int):
        B_free = np.concatenate((np.zeros(d), np.ones(B)))
    else:
        if B_free is None:
            B_free = np.concatenate((np.zeros(d), np.ones(len(B)))) 
        else:
            B_free = np.concatenate((np.zeros(d), B_free))

    model.set_free_params(B_free=B_free, F_free=A2_free, C_free=C1_free, D_free=A1_free)
    model_fitted = model.fit(method=method, bh=bh)
    res = model_fitted.resid
    model_fitted.summary()
    plotACFnPACF(res,noLags=noLags,titleStr=titleStr)
    whiteness_test(res)

    return model_fitted
    
def estimateARMA(y, A=0, C=0, A_free=None, C_free=None, titleStr='', noLags='auto', method='LS', bh=False):
    """Estimates the ARMA model A(z)y(t) = C(z)e(t) using PEM"""
    ordA = A if isinstance(A, int) else len(A)-1
    ordC = C if isinstance(C, int) else len(C)-1
    if ordA == 0 and ordC == 0:
        plotACFnPACF(y, titleStr=titleStr, noLags='auto', includeZeroLag=False)
        return
    
    if not isinstance(A, int): A_free = np.array(A).astype(bool)
    if not isinstance(C, int): C_free = np.array(C).astype(bool)
    model = PEM(y, A=ordA, C=ordC)
    model.set_free_params(A_free=A_free, C_free=C_free)
    model_fitted = model.fit(method=method, bh=bh)
    res = model_fitted.resid
    model_fitted.summary()
    plotACFnPACF(res,noLags=noLags,titleStr=titleStr, includeZeroLag=False)
    whiteness_test(res)

    return model_fitted


def simulateARMA(AR=[1], MA=[1], size=500):
    """
    Simulates an ARMA time series given AR and MA orders or polynomial.

    Parameters:
    - AR (int or list): AR order (if int) or AR polynomial (if list).
    - MA (int or list): MA order (if int) or MA polynomial (if list).
    - size (int, optional): Length of the simulated series. Default is 500.

    Returns:
    - ndarray: Simulated ARMA series.
    - dict (optional): Dictionary containing generated coefficients (if AR or MA is provided as an integer).

    Notes:
    If AR or MA is an integer, stable coefficients are generated.
    """
    coefficients = {}
    # If AR is an integer, generate stable AR coefficients
    if isinstance(AR, int):
        AR = generate_stable_coefficients(AR)
        coefficients['A'] = AR
    
    # If MA is an integer, generate stable MA coefficients
    if isinstance(MA, int):
        MA = generate_stable_coefficients(MA)
        coefficients['C'] = MA

    e = np.random.normal(size=size+100)
    y = filter(MA, AR, e)[100:]
    if coefficients:
        return (y, coefficients)
    else:
        return y

def generate_stable_coefficients(order):
    """Generate stable coefficients for AR or MA."""
    while True:
        # Generate random coefficients
        coefficients = np.concatenate(([1],[np.round(np.random.uniform(-1, 1),2) for _ in range(order)]))
        
        # If coefficients are stable, return them
        roots = np.roots(coefficients[::-1])
        is_stable = all(abs(root) > 1.0 for root in roots)
        if is_stable:
            return coefficients

def simulate_model(x=None, A=[1], B=[0], C=[1], D=[1], F=[1], size=500, e_var=1):
    """
    Simulates a general model of the form A(z)y(t) = [B(z)/F(z)]x(t) + [C(z)/D(z)]e(t).
    """
    if x is None:
        x = np.zeros(size)
    else:
        size = len(x)
    
    # if not len(A): A = [1]
    # if not len(B): B = [0]
    # if not len(C): C = [1]
    # if not len(D): D = [1]
    # if not len(F): F = [1]
    
    e = np.sqrt(e_var)*np.random.normal(size=size+100)
    y = filter(B,np.convolve(A,F),x) + filter(C,np.convolve(A,D), e)[100:]
    return y


def set_order(indices):
    """
    Create an array with 1s at specified indices for time series model order.
    
    Parameters:
    - indices (list): Indices to set to 1.
    
    Returns:
    - np.ndarray: Array with 1s at given indices.
    """
    indices = np.atleast_1d(indices).copy()
    max_index = max(indices)
    result = np.zeros(max_index+1, dtype=int)
    result[indices] = 1
    return result

def equal_length(*args):
    lengths = [len(arg) for arg in args]
    max_length = max(lengths)
    
    out_args = []
    for arg in args:
        out = [0] * max_length
        out[:len(arg)] = arg
        out_args.append(out)
    
    return tuple(out_args)

def polydiv(C, A, k):
    """
    Computes the polynomial division C(z) = A(z)*F(z) + z^{-k}*G(z).
    """
    C,A = equal_length(C,A)
    v = np.concatenate(([1],np.zeros(k-1)))
    F,G = signal.deconvolve(np.convolve(v,C),A)
    return F,G


def recursiveAR(data, order, forgetting_factor=1.0, init_var=1000, theta_guess=None):
    """
    Estimates AR parameters of data using the Recursive Least Squares method.
    """
    
    N = len(data)
    R = np.eye(order) * init_var # Large initial covariance matrix due to uncertainty of guess
    theta = np.zeros(order).reshape(-1, 1) if theta_guess is None else np.array(theta_guess)  # Initial parameters are 0s
    Aest = np.zeros((N, order))
    yhat = np.zeros(N)

    for k in range(order, N):
        x = np.array(data[k-order:k][::-1]).reshape(-1, 1)  # data vector
        # Recursive update
        y = data[k]
        yhat[k] = x.T @ theta
        ehat = y - yhat[k]
        gain = R @ x / (forgetting_factor + x.T @ R @ x)
        theta = theta + gain * ehat
        R = (R - gain @ x.T @ R) / forgetting_factor

        Aest[k, :] = theta.T

    Aest = -Aest

    return Aest, yhat


def recursiveARMA(data, ar_order, ma_order, forgetting_factor=1.0, init_var=1000, theta_guess=None):
    """
    Estimates ARMA parameters of data using the Recursive Least Squares method.
    """
    N = len(data)
    total_order = ar_order + ma_order
    R = np.eye(total_order) * init_var
    theta = np.zeros(total_order).reshape(-1, 1) if theta_guess is None else np.array(theta_guess).reshape(-1, 1)
    ests = np.zeros((N, total_order))
    yhat = np.zeros(N)
    errors = np.zeros(N)

    for k in range(total_order, N):
        x_ar = np.array(data[k-ar_order:k][::-1])
        x_ma = errors[k-ma_order:k][::-1]
        x = np.concatenate((x_ar, x_ma)).reshape(-1, 1)

        y = data[k]
        yhat[k] = x.T @ theta
        ehat = y - yhat[k]
        errors[k] = ehat
        gain = R @ x / (forgetting_factor + x.T @ R @ x)
        theta = theta + gain * ehat
        R = (R - gain @ x.T @ R) / forgetting_factor

        ests[k, :] = theta.T

    # Splitting the estimates for AR and MA parts
    ARest = -ests[:, :ar_order]
    MAest = -ests[:, ar_order:]

    return ARest, MAest, yhat


def predict_pem(PEMResult, y, x=None, k=1, remove=True):
    """
    Computes a k-step prediction of the signal y.

    Parameters:
    - PEMResult (PEMResult): A model fitted using PEM.fit().
    - y (np.ndarray): Endogenous data to use for prediction.
    - x (np.ndarray): Exogenous imput signal. Default is None.
    - k (int): Number of steps for prediction. Default is 1.
    - remove_initial (bool): Whether the fucntion removes the initial corrupt samples. Default is True.

    Returns:
    - y_hat (array-like): The k-step prediction values.
    """
    r = PEMResult
    if r.model.isX: # ARMAX
        # A*y = [B/F]*x + [C/D]*e
        # => A*F*D*y = B*D*x + C*F*e
        K_A = np.convolve(np.convolve(r.A,r.F),r.D)
        K_B = np.convolve(r.B,r.D)
        K_C = np.convolve(r.C,r.F)
        # K_A*y = K_B*x + K_C*e

        F_k, G_k = polydiv(K_C, K_A, k)
        F_khat, G_khat = polydiv(np.convolve(K_B,F_k),K_C,k)
        y_hat = filter(F_khat,1,x) + filter(G_khat,K_C,x) + filter(G_k,K_C,y)
        nr = np.max([len(F_khat),len(G_khat),len(G_k)])
        
    else: # ARMA
        # A*y = [C/D]*e
        # => A*D*y = C*e
        F_k, G_k = polydiv(r.C,np.convolve(r.A,r.D),k)
        y_hat = filter(G_k,r.C,y)
        nr = len(G_k)

    # print('Remember to remove corrupt samples!')
    return y_hat[nr:] if remove else y_hat


def prediction_residual(PEMResult, y, x=None, k=1):
    """
    Calculate the residuals of a k-step prediction for a given signal.

    Parameters:
    - PEMResult (PEMResult): A model fitted using PEM.fit().
    - y (np.ndarray): The original signal data.
    - x (np.ndarray, optional): Exogenous input signal. Default is None.
    - k (int, optional): Number of future steps over which prediction is made. Default is 1.

    Returns:
    - np.ndarray: Residuals from the prediction (actual - predicted values).
    """
    yhat = predict_pem(PEMResult,y,x,k,remove=True)
    res = y[-len(yhat):] - yhat
    return res



def show_model(model): 
    """
    *REPLACED BY PEM summary*
    Constructs and prints the AR and MA polynomials of a ARMA model along with confidence intervals.
    Assumes the model object follows statsmodels SARIMAX/ARIMA/ARMA model structures.
    """
    ar = model.arparams 
    ma = model.maparams

    arpol = model.polynomial_ar
    mapol = model.polynomial_ma 

    confs = model.conf_int()
    confs_ar = -1*confs[1:len(ar)+1]
    confs_ma = confs[::-1][1:len(ma)+1] 

    ar_pairs = list(zip(arpol, [(0, 0)] + confs_ar.tolist()))
    ma_pairs = list(zip(mapol, [(0, 0)] * (len(mapol) - len(ma)) + confs_ma.tolist()))

    ar_string = ' + '.join(
        f'{round(coef, 3)}' + (f' (± {round(abs((c[1] - c[0]) / 2), 3)})' if i > 0 else '') + f'z^-{i}'
        for i, (coef, c) in enumerate(ar_pairs) if coef != 0).replace('z^-0', '')

    ma_string = ' + '.join(
        f'{round(coef, 3)}' + (f' (± {round(abs((c[1] - c[0]) / 2), 3)})' if i > 0 else '') + f'z^-{i}'
        for i, (coef, c) in enumerate(ma_pairs) if coef != 0).replace('z^-0', '')

    print('Discrete-time ARMA model: A(z)y(t) = C(z)e(t)')
    print('A(z) = ',ar_string)
    print('C(z) = ',ma_string)
