import numpy.random as rn
import math
class SimulatedAnnealing(object):

    def __init__(self):
        self.interval = (1.e-4, 1.e-3)


    def clip(self, x):

        a, b = self.interval
        return max(min(x, b), a)

    def random_start(self):
        a, b = self.interval

        return a + (b - a) * rn.random_sample()

    def random_neighbour(self, x, fraction=1):

        #amplitude = (max(self.interval) - min(self.interval)) * fraction / 10

        amplitude = (max(self.interval) - min(self.interval)) * fraction

        delta = (-amplitude/2.) + amplitude * rn.uniform(low=fraction, high=1., size=(1,))[0]

        return self.clip(x + delta)

    def acceptance_probability(self, cost, new_cost, temperature):

        if new_cost < cost:
            return 1.

        else:
            p = math.exp(-(new_cost - cost) / temperature)
            #print("P value =%f", p)
            return p


    #the cooling schedule function

    def temperature(self, c, step):

        return c / math.log(1 + step)

