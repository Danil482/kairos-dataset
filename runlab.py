from solution.pipeline import run
from kairos.data import DATA_DIR

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--final-only", action="store_true")
    args = ap.parse_args()
    do_val = not args.final_only
    do_final = not args.validate_only
    run(args.data, do_validate=do_val, do_final=do_final)