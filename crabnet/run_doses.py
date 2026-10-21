"""Compare 5, 15 and 30 warm-up epochs using same methods and training protocol."""
from pathlib import Path
from run_methods import ROOT, arguments, run


if __name__ == "__main__":
    parser = arguments(__doc__)
    parser.add_argument("--doses", type=int, nargs="+", default=[5, 15, 30])
    parser.add_argument("--output", type=Path, default=ROOT/"runs/doses")
    args = parser.parse_args()
    run(args.doses, args.seeds, args.output, args.device)
