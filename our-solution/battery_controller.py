from numba import jit
import os
from pathlib import Path
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

## set solver params
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 10
# solvers.options['abstol'] = 1e-5
# solvers.options['feastol'] = 1e-6
# solvers.options['reltol'] = 1e-5

class BatteryContoller(object):

    def __init__(self,
                 batt = None, ## battery
                 K = 3, ## number of different scenario
                 E = None, ## error matrix
                 end = 24 * 4 * 10, ## ten days time period
                 T = 4 * 16, ## 10 hours
                 P_E = False, ## compute error matrix
                 i_max = 2): ## the optimization problem is solve every i_max steps
        self.i = i_max
        self.i_max = i_max
        self.first = True
        self.K = K
        self.E = E
        self.horizon = T
        self.end = end
        self.T = min(self.end, self.horizon)
        self.hour = 0
        ## build E error matrices
        if P_E:
            self.build_P_E(dat_df.site_id.unique()[0], dat_df.period_id.unique()[0])


    ## compute error matrix
    def build_P_E(self, site_id,
                  period_id,
                  data_dir = "~/Projects/SE3/data/train/"):
        data_df = pd.read_csv(data_dir + f"{site_id}.csv",
                              parse_dates = ["timestamp"],
                              index_col = "timestamp")

        K = self.K
        H = 24 ## 24 hours
        T = 4 * 24 ## max horizon is 24 hours

        ## compute percentiles
        q = np.linspace(0,1, self.K + 1)
        q = (q[:-1] + q[1:]) / 2

        ## compute error
        E = np.zeros(H * T * K).reshape((K, T, H))
        for t in range(T):
            aux_df = data_df[["actual_consumption", "actual_pv",
                              "load_{0:02}".format(t),
                              "pv_{0:02}".format(t)]].copy()
            aux_df["error"] = (aux_df['actual_consumption'].shift(-t - 1) - aux_df['actual_pv'].shift(-t - 1)) -\
                (aux_df["load_{0:02}".format(t)] - aux_df["pv_{0:02}".format(t)])
            aux_df = aux_df.dropna()
            for h in range(H):
                h_aux_df = aux_df[aux_df.index.hour == h].copy()
                E[:,t,h] = h_aux_df.error.quantile(q).values

        self.E = E

    ## helpers matrices
    @jit
    def build_aux(self):

        ## aux matrice for c computation
        self.aux_c = np.zeros(2 * self.T)
        ### aux matrix: from B+ and B- to B = B+ + B-
        self.aux_B = np.hstack((np.identity(self.T),
                                -np.identity(self.T),
                                np.zeros((self.T,self.T)),
                                np.zeros((self.T,self.T))))
        self.aux_eq_B = np.ones((self.K,1))
        self.aux_eq_G = np.identity(self.K)

        ## create matrices
        self.c = matrix(np.zeros(((self.K + 1) * 2 * self.T)))
        self.A_ub = matrix(np.zeros((2 * self.T * (3 + self.K), 2 * self.T * (1 + self.K))))
        self.b_ub = matrix(np.zeros((2 * self.T * (3 + self.K))))
        self.A_eq = matrix(np.zeros((self.K * self.T, (1 + self.K) * 2 * self.T)))
        self.b_eq = matrix(np.zeros((self.K * self.T)))

    #####################################
    ## A_ub
    @jit
    def build_A_ub_B(self):

        ## capacity limit
        m = np.repeat(1, self.T * self.T).reshape(self.T, self.T)
        m[np.triu_indices(self.T, k = 1)] = 0
        self.A_ub_B = np.vstack((m, -m)) ## no separation
        self.A_ub_B = self.A_ub_B.dot(self.aux_B[:,0:(2 * self.T)])

        ##  battery power limit
        r = 15. / 60.
        aux3 = np.hstack((np.identity(self.T) * (1.0 / r / self.batt.charging_efficiency),
                          np.zeros((self.T,self.T))))
        aux4 = np.hstack((np.zeros((self.T,self.T)),
                          np.identity(self.T) * self.batt.charging_efficiency / r))
        self.A_ub_B = np.vstack((self.A_ub_B,
                                 aux3,
                                 aux4))

    @jit
    def build_A_ub_G(self):
        self.A_ub_G = np.zeros(4 * self.T * (self.T * 2 * self.K)).reshape(4 * self.T,2 * self.T * self.K)

    @jit
    def build_A_ub(self):
        self.build_A_ub_B()
        self.build_A_ub_G()

        ## concatenate matrix for B and G parts
        self.A_ub[0:(4 * self.T),0:(2 * self.T)] = self.A_ub_B
        self.A_ub[0:(4 * self.T),(2 * self.T):((2 * self.T * (1 + self.K)))] = self.A_ub_G

        ## non negativity constraints
        self.A_ub[(4 * self.T):(2 * self.T * (3 + self.K)),:] = -np.identity( 2 * self.T * (1 + self.K))

    #####################################
    ## b_ub
    @jit
    def build_b_ub(self,
                   battery_current_charge):
        ## capacity limit
        self.b_ub[0:self.T] = np.repeat(self.batt.capacity - battery_current_charge, self.T)
        self.b_ub[self.T:(2 * self.T)] = np.repeat(battery_current_charge, self.T)

        ##  battery power limit
        r = 15. / 60. ## because measure are each 15 minutes
        self.b_ub[(2 * self.T):(3 * self.T)] = np.repeat(self.batt.charging_power_limit, self.T)
        self.b_ub[(3 * self.T):(4 * self.T)] = np.repeat(-self.batt.discharging_power_limit, self.T)

        ## non negativity
        self.b_ub[(4 * self.T):((3 + self.K) * 2 * self.T)] = np.zeros(2 * self.T * (1 + self.K))

    #####################################
    ## A_eq
    @jit
    def build_A_eq_B(self):
        ## energy conservation
        self.A_eq_B = np.hstack((-np.identity(self.T) * (1 / self.batt.charging_efficiency),
                                 np.identity(self.T) * (self.batt.discharging_efficiency)))

    @jit
    def build_A_eq_G(self):
        ## energy conservation
        self.A_eq_G = np.hstack((np.identity(self.T),
                                 -np.identity(self.T)))

    @jit
    def build_A_eq(self):

        self.build_A_eq_B()
        self.build_A_eq_G()
        ## concatenate matrix for B and G parts
        self.A_eq[:,0:(2 * self.T)] = np.kron(self.aux_eq_B, self.A_eq_B)
        self.A_eq[:,(2 * self.T):((1 + self.K) * 2 * self.T)] = np.kron(self.aux_eq_G, self.A_eq_G)

    #####################################
    ## b_eq
    @jit
    def build_b_eq(self,
                   load_forecast, pv_forecast):
        self.b_eq[:] = (np.ones((self.K,1)).dot((load_forecast[0:self.T]-pv_forecast[0:self.T]).reshape((1,self.T))) +\
        self.E[:,0:self.T,self.hour]).reshape((self.K * self.T)) ## error

    #####################################
    ## c
    @jit
    def build_c(self,
                price_buy,
                price_sell):
        ## B part
        self.c[0:(2 * self.T)] = np.zeros(2 * self.T)

        ## G part
        self.aux_c[0:self.T] = price_buy[0:self.T]
        self.aux_c[self.T:(2 * self.T)] = -price_sell[0:self.T]
        self.c[(2*self.T):((self.K+1)*2*self.T)] = np.kron(np.ones(self.K), self.aux_c)

    ## solve the optimization problem
    @jit
    def solve(self,
              battery,
              actual_previous_load,
              actual_previous_pv_production,
              price_buy,
              price_sell,
              load_forecast,
              pv_forecast):
        if self.end == 1:
            return 0.0 ## empty the battery at the end of the time period

        if self.i == self.i_max:
            ## update if necessary
            if (self.end < self.horizon):
                self.T = min(self.horizon, self.end)
                self.build_aux()
                self.build_A_ub()
                self.build_A_eq()

            ## create matrices
            self.build_b_ub(battery_current_charge = battery.current_charge * battery.capacity)
            self.build_b_eq(load_forecast.values, pv_forecast.values)
            self.build_c(price_buy.values / 1000., price_sell.values / 1000.) ## because the price is per kWh and energy per Wh

            ## solve LP
            # start = time.clock()
            res = solvers.lp(c = self.c,
                             G = self.A_ub,
                             h = self.b_ub,
                             A = self.A_eq,
                             b = self.b_eq,
                             solver = "gplk")

            ## compute proposed state
            self.x = res["x"][0:self.T] - res["x"][self.T:(2 * self.T)]
            propose_state_of_charge = self.x[0] / battery.capacity + battery.current_charge
            self.i = 1
        else:
            propose_state_of_charge = self.x[self.i] / battery.capacity + battery.current_charge
            self.i = self.i + 1

        ## end - 1
        self.end = self.end - 1

        ## return
        return propose_state_of_charge

    def propose_state_of_charge(self,
                                site_id,
                                timestamp,
                                battery,
                                actual_previous_load,
                                actual_previous_pv_production,
                                price_buy,
                                price_sell,
                                load_forecast,
                                pv_forecast):
        if self.first:## first run
            ## set first to false
            self.first = False
            self.batt = battery
            ## build some useful matrices + init cvxopt matrices
            self.build_aux()
            ## matrices that do not change during the processes
            self.build_A_ub()
            self.build_A_eq()
            ## load E (error) matrix
            assets_path = (Path(__file__)/os.pardir/"assets").resolve()
            if self.K != 1:
                self.E = np.load(assets_path/f"E_{site_id}_{self.K}.pkl")
            else:
                self.E = np.zeros((1,4 * 24, 24))



        ## update hour
        self.hour = timestamp.hour

        return self.solve(battery,
                          actual_previous_load,
                          actual_previous_pv_production,
                          price_buy,
                          price_sell,
                          load_forecast,
                          pv_forecast)
