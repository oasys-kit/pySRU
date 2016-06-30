import numpy as np
import scipy.constants as codata
import scipy.integrate as integrate
from scipy.integrate import odeint
from Trajectory import Trajectory
from UndulatorParameter import UndulatorParameters as Undulator


TRAJECTORY_METHOD_ANALYTIC=0
TRAJECTORY_METHOD_ODE=1
TRAJECTORY_METHOD_INTEGRATION=2


def fct_ODE_plane_undulator(y, t, cst, B):
    return [-cst * B(y[5],y[4]) * y[2],
            0.0,
            cst * B(y[5],y[4]) * y[0],
            y[0],
            0.0,
            y[2]]

def fct_ODE_undulator(y, t, cst, Bx,By,Bz):

    return [cst * (Bz(y[5],y[4]) * y[1] - By(y[5],y[4]) * y[2]),
            cst * (Bx(y[5],y[4]) * y[2] - Bz(y[5],y[4]) * y[0]),
            cst * (By(y[5],y[4]) * y[0] - Bx(y[5],y[4]) * y[1]),
            y[0],
            y[1],
            y[2]]



'''
initial condition : [Vx,Vy,Vz,x,y,z]
'''
class TrajectoryFactory(object):
    def __init__(self,Nb_pts,method,initial_condition=None):
        self.Nb_pts = Nb_pts
        self.method = method
        self.initial_condition=initial_condition

    def copy(self):
        if self.initial_condition==None :
            cond=None
        else :
            cond= self.initial_condition.copy()
        return TrajectoryFactory(Nb_pts=self.Nb_pts,method=self.method,
                                 initial_condition=cond)

    # calculate a theorical trajectory in an undulator
    def analytical_trajectory_plane_undulator(self,undulator):
        N= self.Nb_pts
        ku = 2.0 * np.pi / undulator.lambda_u
        gamma = undulator.E / 0.511e6
        Beta = np.sqrt(1.0 - (1.0 / gamma ** 2))
        Beta_et = 1.0 - (1.0 / (2.0 * gamma ** 2)) * (1.0 + (undulator.K ** 2) / 2.0)
        omega_u = Beta_et * codata.c * ku
        #
        # print("ici")
        # Bo = undulator.K / (93.4 * undulator.lambda_u)
        # print((codata.e*Bo)/(gamma*codata.m_e))
        # print((undulator.K**2*codata.c*omega_u)/(2.0*(gamma**2)*ku))
        # trajectory =
        #   [t........]
        # 	[ X/c......]
        # 	[ Y/c ......]
        #	[ Z/c ......]
        # 	[ Vx/c .....]
        # 	[ Vy/c .....]
        # 	[ Vz/c .....]
        # 	[ Ax/c .....]
        # 	[ Ay/c .....]
        # 	[ Az/c .....]
        trajectory = np.zeros((10, N))
        # t
        trajectory[0] = np.linspace(-undulator.L / (2.0 * codata.c * Beta_et),
                                     undulator.L / (2.0 * codata.c * Beta_et), N)
        #utiliser omegat!
        omegat= np.linspace(-undulator.L *0.5,
                                     undulator.L* 0.5, N)
        omegat *= ku
        # X et Z en fct de t
        trajectory[3] = Beta_et * trajectory[0] + ((undulator.K / gamma) ** 2) * (1.0 / (8.0 * omega_u)) * np.sin(
            2.0 * omegat)
        trajectory[1] = -(undulator.K / (gamma * omega_u)) * np.cos(omegat)
        # Vx et Vz en fct de t
        trajectory[6] = Beta_et + ((undulator.K / gamma) ** 2) * (1.0 / 4.0) * np.cos(2.0 *omegat)
        trajectory[4] = (undulator.K / (gamma )) * np.sin(omegat)
        # Ax et Az en fct de t
        trajectory[9] = -omega_u *( undulator.K / gamma) ** 2 * 0.5 * np.sin(
            2.0 * omegat)
        #trajectory[7] = (undulator.K / (gamma * ku * Beta_et * codata.c)) * (omega_u ** 2) * np.cos(omegat)
        trajectory[7] = (undulator.K / (gamma )) * (omega_u ) * np.cos(omegat)
        return trajectory

    # # a change !!
    def analytical_trajectory_cst_magnf(self,f,t,vz_0,x_0):
        T= np.zeros((10, len(t)))
        T[0]=t
        T[1] = -vz_0 * (1.0 / f) * np.cos(f * t)+vz_0/f + x_0
        T[2] = vz_0 * (0.0) * t
        T[3] = vz_0 * (1.0 / f) * np.sin(f * t)
        T[4] = vz_0 * np.sin(f * t)
        T[5] = vz_0 * 0.0 * t
        T[6] = vz_0 * np.cos(f * t)
        T[7] = vz_0 * f * np.cos(f * t)
        T[8] = vz_0 * 0.0 * t
        T[9] = -vz_0* f * np.sin(f * t)
        return T
    # # a change !!


    # ### cree speciale ment pour un test
    # def analytical_trajectory_cst_magnf(self,Bo,gamma,t,vz_0,x_0):
    #     T=self.copy()
    #     f=Bo*codata.e/(gamma*codata.m_e)
    #     T.x = (lambda t:-vz_0 * (1.0 / f) * np.cos(f * t) +vz_0/f + x_0)
    #     T.y = (lambda t: vz_0 * (0.0) * t)
    #     T.z = (lambda t: vz_0 * (1.0 / f) * np.sin(f * t))
    #     T.v_x = (lambda t: vz_0 * np.sin(f * t))
    #     T.v_y = (lambda t: vz_0 * 0.0 * t)
    #     T.v_z = (lambda t: vz_0 * np.cos(f * t))
    #     T.a_x = (lambda t:vz_0 * f * np.cos(f * t))
    #     T.a_y = (lambda t:vz_0 * 0.0 * t)
    #     T.a_z = (lambda t:-vz_0* f * np.sin(f * t))
    #     return T


    # electron's trajectory in a PLANE undulator that mean :  B=(0,By,0)
    # other hypothesis norm(v)=constant
    def trajectory_undulator_from_magnetic_field_method1(self,undulator, B):
        gamma = undulator.E / 0.511e6
        Beta = np.sqrt(1.0 - (1.0 / gamma ** 2))
        ku = 2.0 * np.pi / undulator.lambda_u
        Beta_et = 1.0 - (1.0 / (2.0 * gamma ** 2)) * (1.0 + (undulator.K ** 2) / 2.0)
        omega_u = Beta_et * codata.c * ku
        N=self.Nb_pts
        Z=np.linspace(B.z[0],B.z[-1],N)
        if type(B.y)== np.ndarray :
            Y = np.linspace(B.y[0], B.y[-1], N)
        else :
            Y=B.y
        # trajectory =
        #   [t........]
        # 	[ X/c......]
        # 	[ Y/c ......]
        #	[ Z/c ......]
        # 	[ Vx/c .....]
        # 	[ Vy/c .....]
        # 	[ Vz/c .....]
        # 	[ Ax/c .....]
        # 	[ Ay/c .....]
        # 	[ Az/c .....]
        trajectory = np.zeros((10, N))
        # t
        trajectory[0] = Z/(Beta_et*codata.c)
        # Ax
        Xm = codata.e *Beta_et/ (gamma * codata.m_e )
        trajectory[7] = Xm * B.By(Z,Y)
        # Vx et Vz
        for i in range(N):
            #np.trapz
            # trajectory[4][i] = integrate.simps(trajectory[7][0:(i + 1)], trajectory[0][0:(i + 1)]) \
            trajectory[4][i] =  integrate.simps(trajectory[7][0:(i + 1)], trajectory[0][0:(i + 1)]) \
                                       + self.initial_condition[0]/ codata.c
        trajectory[6] = np.sqrt((Beta) ** 2 - trajectory[4] ** 2)
        # X et Z
        for i in range(N):
            #trajectory[1][i] = integrate.simps(trajectory[4][0:(i + 1)], trajectory[0][0:(i + 1)]) \
            trajectory[1][i] =  integrate.simps(trajectory[4][0:(i + 1)], trajectory[0][0:(i + 1)]) \
                               + self.initial_condition[3]/ codata.c
            #trajectory[3][i] = integrate.simps(trajectory[6][0:(i + 1)], trajectory[0][0:(i + 1)]) \
            trajectory[3][i] =  integrate.simps(trajectory[6][0:(i + 1)], trajectory[0][0:(i + 1)]) \
                               +  self.initial_condition[5]/ codata.c

            # Az
        trajectory[9] = -(trajectory[7] * trajectory[4]) / trajectory[6]

        return trajectory

    # electron's trajectory in a PLANE undulator that mean :  B=(0,By,0)
    def trajectory_undulator_from_magnetic_field_method2(self,undulator, B):
        gamma = undulator.gamma()
        Z = B.z
        Beta = np.sqrt(1.0 - (1.0 / gamma ** 2))
        Beta_et = 1.0 - (1.0 / (2.0 * gamma ** 2)) * (1.0 + (undulator.K ** 2) / 2.0)
        N=self.Nb_pts
        #   trajectory =
        #   [t........]
        # 	[ X/c......]
        # 	[ Y/c ......]
        #	[ Z/c ......]
        # 	[ Vx/c .....]
        # 	[ Vy/c .....]
        # 	[ Vz/c .....]
        # 	[ Ax/c .....]
        # 	[ Ay/c .....]
        # 	[ Az/c .....]
        trajectory = np.zeros((10, N))
        trajectory[0] = np.linspace(Z[0] / (Beta_et * codata.c), Z[- 1] / (Beta_et * codata.c), N)
        cst = -codata.e / (codata.m_e * gamma)
        res = odeint(fct_ODE_undulator,self.initial_condition, trajectory[0],
                     args=(cst,B.Bx,B.By,B.Bz), full_output=True)
        traj = res[0]
        info = res[1]
        # print("1 : nonstiff problems, Adams . 2: stiff problem, BDF")
        # print(info.get('mused'))
        traj = np.transpose(traj)
        trajectory[4] = traj[0]
        trajectory[5] = traj[1]
        trajectory[6] = traj[2]
        trajectory[1] = traj[3]
        trajectory[2] = traj[4]
        trajectory[3] = traj[5]
        trajectory[7] = -cst * B.By(trajectory[3],trajectory[2]) * trajectory[6]
        trajectory[9] = cst * B.By(trajectory[3],trajectory[2]) * trajectory[4]
        k = 1
        while k < 10:
            trajectory[k] *= 1.0 / codata.c
            k += 1
        return trajectory

    def calculate_trajectory(self,undulator,B):
        if (self.method == TRAJECTORY_METHOD_ODE or self.method == TRAJECTORY_METHOD_INTEGRATION):
            if self.method == TRAJECTORY_METHOD_INTEGRATION:
                T = self.trajectory_undulator_from_magnetic_field_method1(undulator=undulator, B=B)


            else: # method=TRAJECTORY_METHOD_ODE
                T = self.trajectory_undulator_from_magnetic_field_method2(undulator=undulator, B=B)

        else:
            T = self.analytical_trajectory_plane_undulator(undulator=undulator)
        return T

    def create_for_plane_undulator(self,undulator,B):
        if (self.method == 1 or self.method == 2):
            if (self.initial_condition==None) :
                self.initial_condition=np.array([0.0,0.0,np.sqrt(1.0 - (1.0 / ( undulator.E /0.511e6)** 2))*codata.c,
                                                 0.0,0.0,B.z[0]])
                #print(self.initial_condition)
            T=self.calculate_trajectory(undulator=undulator,B=B)

        else:
            T = self.analytical_trajectory_plane_undulator(undulator=undulator)

        trajectory = Trajectory(T[0],T[1],T[2],T[3],T[4],T[5],T[6],T[7],T[8],T[9])

        return trajectory

    def create_for_plane_undulator_ideal(self,undulator):
        T = self.analytical_trajectory_plane_undulator(undulator)
        trajectory = Trajectory(T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9])
        self.initial_condition=np.array([T[4][0],T[5][0],T[6][0],T[1][0],T[2][0],T[3][0]])
        self.method=TRAJECTORY_METHOD_ANALYTIC
        return trajectory

    def create_for_cst_magnetic_field(self, undulator,t):
        Bo = (undulator.K / (93.4 * undulator.lambda_u))
        f = Bo * codata.e / (undulator.gamma() * codata.m_e)
        T = self.analytical_trajectory_cst_magnf(f,t=t,vz_0=undulator.Beta(),
                                                 x_0=-undulator.Beta()/f)
        trajectory = Trajectory(T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9])
        return trajectory





if __name__ == "__main__" :
    und_test = Undulator(K=1.87, E=1.3e9, lambda_u=0.035, L=0.035 * 12, I=1.0)
    traj_test=TrajectoryFactory(Nb_pts=201,method=TRAJECTORY_METHOD_ANALYTIC).create_for_plane_undulator_ideal(
                                                                                                undulator=und_test)
    Beta_et = 1.0 - (1.0 / (2.0 * und_test.gamma() ** 2)) * (1.0 + (und_test.K ** 2) / 2.0)

    ku=2.0*np.pi/und_test.lambda_u

    xo = und_test.K / (und_test.gamma() * Beta_et * ku)
    zo=Beta_et*codata.c*und_test.L / (2.0 * codata.c * Beta_et)
    vzo=Beta_et*codata.c - ((und_test.K / und_test.gamma()) ** 2)

    initial_condition=np.array([0.0,0.0,vzo,xo,0.0,zo])
    traj_test2 = TrajectoryFactory(Nb_pts=201, method=TRAJECTORY_METHOD_INTEGRATION,
                                   initial_condition=initial_condition).create_for_plane_undulator_ideal(
                                    undulator=und_test)

    traj_test.draw_2_trajectory(traj_test2)
    print(traj_test.z.max())