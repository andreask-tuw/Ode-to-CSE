import math


import sys
# sys.path.append('/Users/joachim/texjs/lva/IntroSC/ASC-ODE/build/mechsystem')
sys.path.append('../build/mechsystem')

from mass_spring import *

mss = MassSpringSystem3d()
mss.gravity = (0,0,-9.81)

mA = mss.add (Mass(1, (1,0,0)))
mB = mss.add (Mass(2, (2,0,0)))
f1 = mss.add (Fix( (0,0,0)) )
mss.add (Spring(1, 10, (f1, mA)))
mss.add (Spring(1, 20, (mA, mB)))

mss.add(DistanceConstraint(mA, mB, 1.0))


print("nj =", len(mss.joints))
state = mss.getState()
print("state len =", len(state))
print("state =", state)

def print_dist(tag):
    s = mss.getState()
    pA = s[0:3]
    pB = s[3:6]
    d = math.sqrt((pA[0]-pB[0])**2 + (pA[1]-pB[1])**2 + (pA[2]-pB[2])**2)
    print(tag, "dist =", d, "lambdas =", s[6:])

print_dist("initial")

mss.simulate(0.1, 10)
print_dist("after 1")

mss.simulate(0.1, 10)
print_dist("after 2")

print ("state = ", mss.getState())

mss.simulate (0.1, 10)

print ("state = ", mss.getState())


mss.simulate (0.1, 10)

print ("state = ", mss.getState())

for m in mss.masses:
    print (m.mass, m.pos)

mss.masses[0].mass = 5

for m in mss.masses:
    print (m.mass, m.pos)


def print_dist(tag):
    s = mss.getState()     
    pA = s[0:3]
    pB = s[3:6]
    d = math.dist(pA, pB)
    print(tag, "dist =", d)



tend, steps = 0.1, 10
full = mss.simulate(tend, steps)

nm = len(mss.masses)
n_mass = 3 * nm
nj = len(mss.joints)

print("lambdas after 1 =", full[n_mass:n_mass+nj])


full = mss.simulate(tend, steps)
print_dist("after 2")
print("lambdas after 2 =", full[n_mass:n_mass+nj])
