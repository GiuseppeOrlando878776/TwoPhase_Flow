from My_Parameters import My_Parameters

def main():
    param_test = My_Parameters("test.cfg")
    param_handler = param_test.get_param()
    try:
        print(str(param_handler["Raylegh_number"]))
    except RuntimeError as e:
        print(str(e) +  "\nPlease check configuration file")

if __name__ == "__main__":
    main()
