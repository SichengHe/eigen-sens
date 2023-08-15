import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib

# color
my_blue = '#4C72B0'
my_red = '#C54E52'
my_green = '#56A968' 
my_brown = '#b4943e'
my_purple = '#684c6b'
my_orange = '#cc5500'

# font
matplotlib.rcParams.update({'font.size': 20})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


# spectral differentiation
def evaluateList(f, x_list):

	y_list = []

	for i in range(len(x_list)):

		y_list.append(f(x_list[i]))

	return y_list


class TSinter(object):

	# we assume we have odd time instances

	def __init__(self, u):

		self.u = u

		self.ntimeinstances = len(u)

		ntimeinstances = self.ntimeinstances

		if (ntimeinstances%2 == 0):
			print("err: even time instances")
			exit()

		self.mode_N = (ntimeinstances - 1) / 2


	def interpolate(self):

		u = self.u
		mode_N = self.mode_N
		ntimeinstances = self.ntimeinstances

		uf = np.fft.fft(u)

		urf = []

		for i in range(int(mode_N + 1)):

			if i==0:

				urf.append([np.real(uf[0]/ ntimeinstances)])

			else:

				Cc = ( np.real(uf[i]) + np.real(uf[-i])) / ntimeinstances
				Cs = (-np.imag(uf[i]) + np.imag(uf[-i])) / ntimeinstances

				mag = np.sqrt(Cc**2 + Cs**2) 

				phase = np.arcsin(Cc/mag)


				urf.append([mag, phase])

		self.urf = urf

	def __call__(self, phase_loc):

		u_loc = self.computeOnePoint(phase_loc)

		return u_loc
	
	def computeOnePoint(self, phase_loc):

		mode_N = self.mode_N
		urf = self.urf

		u_loc = 0.0

		for i in range(int(mode_N + 1)):

			if i==0:

				u_loc += urf[0][0]

			else:

                
				mag, phase = urf[i]

				u_loc += mag*np.sin(phase + phase_loc)

		return u_loc
	
	def compute(self, N, isNegPhase = False):

		if isNegPhase:
			phase_list = [- (2 * np.pi / N * i) for i in range(N)]
		else:
			phase_list = [(2 * np.pi / N * i) for i in range(N)]

		u_list = [self.computeOnePoint(x) for x in phase_list]

		return u_list


# regression test
flag_regtest = 0
# f(phase) = 1.5 sin(phase + pi/4)
if flag_regtest:

	u_reg = []

	for i in range(3):

		phase = (np.float(i)/3)*(2.0*np.pi)

		u_reg.append(1.5*np.sin(phase + np.pi/4))

	TSobj = TSinter(u_reg)
	TSobj.interpolate()
	# f(0) = 1.5 sin(pi/4) 1.0606601717798212
	print(TSobj(0.0))
	# f(pi/3) = 1.5 sin(7pi/12) 1.4488887394336025
	print(TSobj(np.pi/3))
	# f(2pi/3) = 1.5 sin(11pi/12) 0.38822856765378155
	print(TSobj(np.pi/3*2.0))


def richardsonExtra(fValue, N, d = 2, is2ndOrder = True):

	"""
		Conduct Richardson extrapolation given value list "fValue" and mesh size list "N".
		d is the dimension.
	"""

	fValue_L0, fValue_L1, fValue_L2 = fValue

	h_L0, h_L1, h_L2 = [x**(-1.0 / d) for x in N]

	r = h_L1 / h_L0

	if (not is2ndOrder):
		p = np.log((fValue_L2 - fValue_L1) / (fValue_L1 - fValue_L0)) / np.log(r)
		print(p)
	else:
		p = 2

	f_h0 = fValue_L0 + (fValue_L0 - fValue_L1) / (r**p - 1.0)


	return f_h0


def richardsonExtraList(fValueList, N, d=2, is2ndOrder = True):

	"""
		Conduct Richardson extrapolation given value a list of coarse, medium and fine value lists.
	"""

	fValue_L0_list, fValue_L1_list, fValue_L2_list = fValueList
	print("fValue_L0_list", fValue_L0_list)

	NData = len(fValue_L0_list)

	f_h0_list = []

	for i in range(NData):

		fValue_L0_loc = fValue_L0_list[i]
		fValue_L1_loc = fValue_L1_list[i]
		fValue_L2_loc = fValue_L2_list[i]

		fValue_loc = [fValue_L0_loc, fValue_L1_loc, fValue_L2_loc]

		f_h0_loc = richardsonExtra(fValue_loc, N, d = 2, is2ndOrder = is2ndOrder)

		f_h0_list.append(f_h0_loc)

	return f_h0_list


	



