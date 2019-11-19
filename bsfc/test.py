import argparse




parser = argparse.ArgumentParser()
parser.add_argument("shot", type=int, help="shot number to run analysis on")
parser.add_argument("-l", "--line_name", help="name of atomic line of interest for post-fitting analysis. For the primary line, on can just leave to None")
parser.add_argument('-f', "--force", action="store_true", help="whether or not to force an overwrite of saved data")



args = parser.parse_args()


print args
