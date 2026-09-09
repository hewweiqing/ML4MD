"""ALIGNN at 5, 15 and 30 warm-up passes."""
from run_methods import parser, run


if __name__ == "__main__":
    p = parser(__doc__, output="doses")
    p.add_argument("--doses", nargs="+", type=int, default=[5, 15, 30])
    args = p.parse_args()
    run(args, args.doses)
