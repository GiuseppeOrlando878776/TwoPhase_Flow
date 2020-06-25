from sys import argv
from Bubble_move import *
from Rayleigh_Taylor import *

def main():
    if(len(argv) >= 2):
        param_test = My_Parameters(argv[1])
    else:
        param_test = My_Parameters("test_RT.cfg")
    param_handler = param_test.get_param()
    try:
        print(str(param_handler["Reinit_Type"]))
        print(param_handler["Stabilization_Type"])
        print(param_handler["NS_Procedure"])
    except RuntimeError as e:
        print(str(e) +  "\nPlease check configuration file")

    sim = RayleighTaylor("test_RT.cfg")
    sim.run()


if __name__ == "__main__":
    main()
