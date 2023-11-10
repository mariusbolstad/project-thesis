Daily data:


Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=29103.382, Time=1.67 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=29881.406, Time=0.03 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=29109.394, Time=0.07 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=29232.123, Time=0.28 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=29879.406, Time=0.02 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=29105.314, Time=0.29 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=29107.815, Time=1.05 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=29101.033, Time=1.41 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=29102.039, Time=0.72 sec
 ARIMA(4,1,2)(0,0,0)[0] intercept   : AIC=29106.012, Time=0.62 sec
 ARIMA(3,1,3)(0,0,0)[0] intercept   : AIC=29080.575, Time=2.52 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=29095.234, Time=2.14 sec
 ARIMA(4,1,3)(0,0,0)[0] intercept   : AIC=29103.720, Time=2.64 sec
 ARIMA(3,1,4)(0,0,0)[0] intercept   : AIC=29100.723, Time=2.69 sec
 ARIMA(2,1,4)(0,0,0)[0] intercept   : AIC=29104.993, Time=1.32 sec
 ARIMA(4,1,4)(0,0,0)[0] intercept   : AIC=29099.823, Time=3.15 sec
 ARIMA(3,1,3)(0,0,0)[0]             : AIC=29081.497, Time=1.32 sec

Best model:  ARIMA(3,1,3)(0,0,0)[0] intercept
Total fit time: 21.946 seconds
SARIMAX Results
Dep. Variable:	y	No. Observations:	2346
Model:	SARIMAX(3, 1, 3)	Log Likelihood	-14532.288
Date:	Thu, 09 Nov 2023	AIC	29080.575
Time:	14:59:39	BIC	29126.656
Sample:	0	HQIC	29097.359
- 2346		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
intercept	0.0606	0.086	0.707	0.480	-0.107	0.229
ar.L1	2.1379	0.054	39.310	0.000	2.031	2.245
ar.L2	-1.3651	0.095	-14.438	0.000	-1.550	-1.180
ar.L3	0.2094	0.041	5.055	0.000	0.128	0.291
ma.L1	-1.5923	0.054	-29.324	0.000	-1.699	-1.486
ma.L2	0.4039	0.073	5.541	0.000	0.261	0.547
ma.L3	0.2199	0.025	8.685	0.000	0.170	0.269
sigma2	1.417e+04	175.720	80.665	0.000	1.38e+04	1.45e+04
Ljung-Box (L1) (Q):	0.12	Jarque-Bera (JB):	17891.72
Prob(Q):	0.73	Prob(JB):	0.00
Heteroskedasticity (H):	4.22	Skew:	0.78
Prob(H) (two-sided):	0.00	Kurtosis:	16.44


                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                 1876
Model:                 ARIMA(3, 1, 3)   Log Likelihood              -11456.363
Date:                Thu, 09 Nov 2023   AIC                          22926.727
Time:                        15:03:29   BIC                          22965.481
Sample:                             0   HQIC                         22941.003
                               - 1876                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.4951      0.062     -7.944      0.000      -0.617      -0.373
ar.L2         -0.3056      0.057     -5.341      0.000      -0.418      -0.193
ar.L3          0.4723      0.033     14.209      0.000       0.407       0.537
ma.L1          1.0592      0.065     16.343      0.000       0.932       1.186
ma.L2          0.8791      0.078     11.274      0.000       0.726       1.032
ma.L3         -0.0167      0.027     -0.627      0.530      -0.069       0.035
sigma2      1.189e+04    158.713     74.884      0.000    1.16e+04    1.22e+04
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             20876.38
Prob(Q):                              0.98   Prob(JB):                         0.00
Heteroskedasticity (H):               3.49   Skew:                             1.07
Prob(H) (two-sided):                  0.00   Kurtosis:                        19.21
===================================================================================


                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                 1877
Model:                 ARIMA(3, 1, 3)   Log Likelihood              -11461.227
Date:                Thu, 09 Nov 2023   AIC                          22936.455
Time:                        15:03:32   BIC                          22975.213
Sample:                             0   HQIC                         22950.732
                               - 1877                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.5502      0.042    -13.175      0.000      -0.632      -0.468
ar.L2         -0.3311      0.037     -8.968      0.000      -0.404      -0.259
ar.L3          0.4867      0.023     20.993      0.000       0.441       0.532
ma.L1          1.1143      0.043     25.703      0.000       1.029       1.199
ma.L2          0.9385      0.050     18.759      0.000       0.840       1.037
ma.L3         -0.0043      0.026     -0.164      0.870      -0.056       0.047
sigma2      1.188e+04    156.834     75.749      0.000    1.16e+04    1.22e+04
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):             20443.72
Prob(Q):                              1.00   Prob(JB):                         0.00
Heteroskedasticity (H):               3.49   Skew:                             1.07
Prob(H) (two-sided):                  0.00   Kurtosis:                        19.03
===================================================================================

Weekly data:

Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=7817.549, Time=0.36 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=7883.694, Time=0.04 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=7839.387, Time=0.06 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=7844.652, Time=0.06 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=7881.694, Time=0.05 sec
 ARIMA(1,1,2)(0,0,0)[0] intercept   : AIC=7831.004, Time=0.19 sec
 ARIMA(2,1,1)(0,0,0)[0] intercept   : AIC=7839.305, Time=0.32 sec
 ARIMA(3,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.49 sec
 ARIMA(2,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.60 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=7841.257, Time=0.09 sec
 ARIMA(1,1,3)(0,0,0)[0] intercept   : AIC=inf, Time=0.57 sec
 ARIMA(3,1,1)(0,0,0)[0] intercept   : AIC=7816.198, Time=0.32 sec
 ARIMA(3,1,0)(0,0,0)[0] intercept   : AIC=7821.903, Time=0.10 sec
 ARIMA(4,1,1)(0,0,0)[0] intercept   : AIC=inf, Time=0.65 sec
 ARIMA(2,1,0)(0,0,0)[0] intercept   : AIC=7841.087, Time=0.06 sec
 ARIMA(4,1,0)(0,0,0)[0] intercept   : AIC=7822.069, Time=0.10 sec
 ARIMA(4,1,2)(0,0,0)[0] intercept   : AIC=7818.038, Time=0.34 sec
 ARIMA(3,1,1)(0,0,0)[0]             : AIC=7814.254, Time=0.14 sec
 ARIMA(2,1,1)(0,0,0)[0]             : AIC=7837.304, Time=0.18 sec
 ARIMA(3,1,0)(0,0,0)[0]             : AIC=7819.900, Time=0.06 sec
 ARIMA(4,1,1)(0,0,0)[0]             : AIC=inf, Time=0.45 sec
 ARIMA(3,1,2)(0,0,0)[0]             : AIC=inf, Time=0.18 sec
 ARIMA(2,1,0)(0,0,0)[0]             : AIC=7839.076, Time=0.06 sec
 ARIMA(2,1,2)(0,0,0)[0]             : AIC=7815.548, Time=0.20 sec
...
 ARIMA(4,1,2)(0,0,0)[0]             : AIC=7816.038, Time=0.19 sec

Best model:  ARIMA(3,1,1)(0,0,0)[0]          
Total fit time: 5.922 seconds


SARIMAX Results
Dep. Variable:	y	No. Observations:	524
Model:	SARIMAX(3, 1, 1)	Log Likelihood	-3902.127
Date:	Thu, 09 Nov 2023	AIC	7814.254
Time:	15:39:36	BIC	7835.552
Sample:	10-20-2013	HQIC	7822.596
- 10-29-2023		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
ar.L1	0.8449	0.108	7.819	0.000	0.633	1.057
ar.L2	-0.1306	0.064	-2.027	0.043	-0.257	-0.004
ar.L3	-0.1778	0.043	-4.151	0.000	-0.262	-0.094
ma.L1	-0.5823	0.102	-5.730	0.000	-0.782	-0.383
sigma2	1.795e+05	7000.509	25.640	0.000	1.66e+05	1.93e+05
Ljung-Box (L1) (Q):	0.02	Jarque-Bera (JB):	460.58
Prob(Q):	0.90	Prob(JB):	0.00
Heteroskedasticity (H):	2.87	Skew:	-0.39
Prob(H) (two-sided):	0.00	Kurtosis:	7.53


                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                  419
Model:                 ARIMA(3, 1, 1)   Log Likelihood               -3094.776
Date:                Thu, 09 Nov 2023   AIC                           6199.553
Time:                        15:40:59   BIC                           6219.730
Sample:                    10-20-2013   HQIC                          6207.529
                         - 10-24-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.8339      0.151      5.523      0.000       0.538       1.130
ar.L2         -0.1755      0.080     -2.196      0.028      -0.332      -0.019
ar.L3         -0.1704      0.055     -3.123      0.002      -0.277      -0.063
ma.L1         -0.5277      0.145     -3.641      0.000      -0.812      -0.244
sigma2       1.59e+05   7180.714     22.147      0.000    1.45e+05    1.73e+05
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):               360.25
Prob(Q):                              0.90   Prob(JB):                         0.00
Heteroskedasticity (H):               2.50   Skew:                            -0.01
Prob(H) (two-sided):                  0.00   Kurtosis:                         7.55
===================================================================================




                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                  419
Model:                 ARIMA(3, 1, 1)   Log Likelihood               -3091.158
Date:                Thu, 09 Nov 2023   AIC                           6196.317
Time:                        15:40:59   BIC                           6224.565
Sample:                    10-20-2013   HQIC                          6207.484
                         - 10-24-2021                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
IRON_CLOSE    -0.6891      0.485     -1.421      0.155      -1.640       0.261
COAL_CLOSE     0.5028      0.198      2.539      0.011       0.115       0.891
ar.L1          0.8241      0.146      5.662      0.000       0.539       1.109
ar.L2         -0.1595      0.079     -2.007      0.045      -0.315      -0.004
ar.L3         -0.1858      0.057     -3.264      0.001      -0.297      -0.074
ma.L1         -0.5382      0.145     -3.717      0.000      -0.822      -0.254
sigma2      1.581e+05   7369.694     21.449      0.000    1.44e+05    1.73e+05
===================================================================================
Ljung-Box (L1) (Q):                   0.02   Jarque-Bera (JB):               499.51
Prob(Q):                              0.89   Prob(JB):                         0.00
Heteroskedasticity (H):               2.38   Skew:                            -0.17
Prob(H) (two-sided):                  0.00   Kurtosis:                         8.34
===================================================================================


Monthly data:

Performing stepwise search to minimize aic
 ARIMA(2,1,2)(0,0,0)[0] intercept   : AIC=inf, Time=0.32 sec
 ARIMA(0,1,0)(0,0,0)[0] intercept   : AIC=4896.516, Time=0.02 sec
 ARIMA(1,1,0)(0,0,0)[0] intercept   : AIC=4898.378, Time=0.03 sec
 ARIMA(0,1,1)(0,0,0)[0] intercept   : AIC=4898.392, Time=0.03 sec
 ARIMA(0,1,0)(0,0,0)[0]             : AIC=4894.518, Time=0.02 sec
 ARIMA(1,1,1)(0,0,0)[0] intercept   : AIC=4894.734, Time=0.13 sec

Best model:  ARIMA(0,1,0)(0,0,0)[0]          
Total fit time: 0.561 seconds
SARIMAX Results
Dep. Variable:	y	No. Observations:	289
Model:	SARIMAX(0, 1, 0)	Log Likelihood	-2446.259
Date:	Thu, 09 Nov 2023	AIC	4894.518
Time:	15:43:43	BIC	4898.181
Sample:	03-31-1999	HQIC	4895.986
- 03-31-2023		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
sigma2	1.392e+06	5.91e+04	23.544	0.000	1.28e+06	1.51e+06
Ljung-Box (L1) (Q):	0.14	Jarque-Bera (JB):	406.72
Prob(Q):	0.71	Prob(JB):	0.00
Heteroskedasticity (H):	3.07	Skew:	-0.70
Prob(H) (two-sided):	0.00	Kurtosis:	8.65

                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                  231
Model:                 ARIMA(1, 1, 1)   Log Likelihood               -1937.949
Date:                Thu, 09 Nov 2023   AIC                           3881.898
Time:                        15:44:41   BIC                           3892.213
Sample:                    03-31-1999   HQIC                          3886.059
                         - 05-31-2018                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1         -0.7468      0.055    -13.680      0.000      -0.854      -0.640
ma.L1          0.9082      0.044     20.575      0.000       0.822       0.995
sigma2      1.218e+06   6.81e+04     17.881      0.000    1.08e+06    1.35e+06
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):               280.46
Prob(Q):                              0.93   Prob(JB):                         0.00
Heteroskedasticity (H):               1.88   Skew:                            -0.87
Prob(H) (two-sided):                  0.01   Kurtosis:                         8.12
===================================================================================


                               SARIMAX Results                                
==============================================================================
Dep. Variable:                  CLOSE   No. Observations:                  231
Model:                 ARIMA(1, 1, 1)   Log Likelihood               -1936.959
Date:                Thu, 09 Nov 2023   AIC                           3883.917
Time:                        15:44:41   BIC                           3901.108
Sample:                    03-31-1999   HQIC                          3890.851
                         - 05-31-2018                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
IRON_CLOSE    -0.3638      0.725     -0.502      0.616      -1.784       1.057
COAL_CLOSE     0.1870      0.080      2.347      0.019       0.031       0.343
ar.L1         -0.7622      0.050    -15.306      0.000      -0.860      -0.665
ma.L1          0.9335      0.033     28.215      0.000       0.869       0.998
sigma2      1.207e+06   6.85e+04     17.617      0.000    1.07e+06    1.34e+06
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               310.86
Prob(Q):                              0.99   Prob(JB):                         0.00
Heteroskedasticity (H):               1.98   Skew:                            -0.95
Prob(H) (two-sided):                  0.00   Kurtosis:                         8.37
===================================================================================