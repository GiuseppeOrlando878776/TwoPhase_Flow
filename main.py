from sys import argv
from Bubble_move import *
from Rayleigh_Taylor import *

def main():
    if(len(argv) == 2):
        config_file = argv[1]
    elif(len(argv) == 1):
        config_file = "test.cfg"
    else:
        raise RuntimeError("Wrong number of arguments in the call")

    #Build the parameters handler
    param_handler = My_Parameters(config_file).get_param()

    #Build the right problem
    if(param_handler["Problem"] == 'Bubble'):
        sim = BubbleMove(param_handler)
        #Save also the communicator for future printing purporses
        comm = sim.get_communicator()
    elif(param_handler["Problem"] == 'RT'):
        sim = RayleighTaylor(param_handler)
        #Save also the communicator for future printing purporses
        comm = sim.get_communicator()
    else:
        raise ValueError("Unknown problem type. Please check configuration file")

    #Run the simulation
    try:
        sim.run()
    except ValueError as e:
        if(MPI.rank(comm) == 0):
            print(e)
            print("Aborting simulation...")
        exit(1)


if __name__ == "__main__":
    main()
