import xarray as xr
import scipy.stats as stats
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ds = xr.open_dataset(args.input_path)

    ## computing skews
    print("Computing skews...")
    ds_skew = ds.reduce(stats.skew, dim="time")
    ds_skew = ds_skew.mean(dim=["lat", "lon"])

    ## creating skew map
    ds_skew_map = ds_skew.to_pandas()
    ds_skew_map.name = "skew"

    ## creating reduction map
    reduction_map = dict()

    for var, value in ds_skew_map.items():
        if abs(value) < 1:
            reduction_map[var] = lambda x: x.mean(dim="time")
        else:
            reduction_map[var] = lambda x: x.median(dim="time")


    # 2. Reconstruct the dataset with reduced variables
    print("Reconstructing reduced dataset...")
    ds_reduced = xr.Dataset(
        {var_name: func(ds[var_name]) for var_name, func in reduction_map.items()},
        attrs=ds.attrs
    )
    
    ds_reduced.to_netcdf(args.output_path)
    
    print("Done!")
