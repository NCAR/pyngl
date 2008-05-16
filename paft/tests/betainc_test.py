import Ngl,types

#
# Leave these both alone, regardless of what module you are testing.
#
import numpy
from utils import *

#
# Begin betainc tests.
#
x = 0.2
a = 0.5
b = 5.0
alpha1 = Ngl.betainc(x,a,b) 

x = 0.5
alpha2 = Ngl.betainc(x,a,b) 

#
# Correct values for first set of tests.
#
alpha1_value  = 0.855072394596
alpha2_value = 0.989880440265

test_value("betainc",alpha1,alpha1_value)
test_value("betainc",alpha2,alpha2_value)
check_type(alpha1)
check_type(alpha2)

# Test arrays of values.

x = [0.2,0.2,0.2,0.2]
a = [0.5,0.5,0.5,0.5]
b = [5.0,5.0,5.0,5.0]
alpha1_value  = [0.855072394596,0.855072394596,0.855072394596,0.855072394596]
alpha1 = Ngl.betainc(x,a,b) 

test_values("betainc (list)",alpha1,alpha1_value)
check_type(alpha1)

x = numpy.array([0.2,0.2,0.2,0.2])
a = [0.5,0.5,0.5,0.5]
b = [5.0,5.0,5.0,5.0]
alpha1_value  = [0.855072394596,0.855072394596,0.855072394596,0.855072394596]
alpha1 = Ngl.betainc(x,a,b) 

test_values("betainc (numpy array)",alpha1,alpha1_value)
check_type(alpha1)

a = numpy.array([0.5,0.5,0.5,0.5])
b = numpy.array([5.0,5.0,5.0,5.0])
test_values("betainc (numpy array)",alpha1,alpha1_value)
check_type(alpha1)

#
# More betainc tests
#
df   = 20
tval = 2.09
prob = Ngl.betainc(df/(df+tval**2), df/2.0, 0.5)

prob_value = 0.04959886483
test_value("betainc",prob,prob_value)
check_type(prob)

