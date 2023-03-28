import os

from needle_shape_sensing.sensorized_needles import MCFNeedle
from needle_shape_sensing.shape_sensing import ShapeSensingMCFNeedle

def test_ssmcfneedle():
    mcfneedle = MCFNeedle.load_json(os.path.join(
        "data",
        "needle_params_MCF-4CH-4AA-0001.json"
    ))

    ss_mcfneedle = ShapeSensingMCFNeedle.from_MCFNeedle(mcfneedle)

    print(repr(ss_mcfneedle))


def main():
    fns = [
        test_ssmcfneedle
    ]

    for fn in fns:
        print("Testing:", fn.__qualname__)
        fn()

    # for

# main

if __name__ == "__main__":
    main()

# if __main__