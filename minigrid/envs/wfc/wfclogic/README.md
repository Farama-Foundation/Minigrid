# wfc_2019f

This is my research implementation of WaveFunctionCollapse in Python. It has two goals:

* Make it easier to understand how the algorithm operates
* Provide a testbed for experimenting with alternate heuristics and features

For more general-purpose WFC information, the original reference repository remains the best resource: https://github.com/mxgmn/WaveFunctionCollapse

## Running WFC

If you want direct control over running WFC, call `wfc_control.execute_wfc()`.

The arguments it accepts are:

- `tile_size=1`: size of the tiles it uses (1 is fine for pixel images, larger is for things like a Super Metroid map)
- `pattern_width=2`: size of the patterns; usually 2 or 3 because bigger gets slower and
- `rotations=8`: how many reflections and/or rotations to use with the patterns
- `output_size=[48,48]`: how big the output image is
- `ground=None`: which patterns should be placed along the bottom-most line
- `attempt_limit=10`: stop after this many tries
- `output_periodic=True`: the output wraps at the edges
- `input_periodic=True`: the input wraps at the edges
- `loc_heuristic="entropy"`: what location heuristic to use; `entropy` is the original WFC behavior. The heuristics that are implemented are `lexical`, `hilbert`, `spiral`, `entropy`, `anti-entropy`, `simple`, `random`, but when in doubt stick with `entropy`.
- `choice_heuristic="weighted"`: what choice heuristic to use; `weighted` is the original WFC behavior, other options are `random`, `rarest`, and `lexical`.
- `global_constraint=False`: what global constraint to use. Currently the only one implemented is `allpatterns`
- `backtracking=False`: do we use backtracking if we run into a contradiction?
- `log_filename="log"`: what should the log file be named?
- `logging=False`: should we write to a log file?  requires `filename`.
- `log_stats_to_output=None`
- `image`: an array of pixel data, typically in the shape: (height, width, rgb)

## Test

```
pytest
```

## Documentation

```
python setup.py build_sphinx
```

With linux the documentation can be displayed with:

```
xdg-open build/sphinx/index.html
```
