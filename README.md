# Automata README
A collection of my Automata / art coding hobbies. I have set it up such that relevant parameters are read from a parameters.py file.

I appreciate any constructive feedback or future ideas!

## The Walker and Canvas classes
Uses the 'canvas' and 'walker' parameters to simulate restricted walkers on a canvas.

```bash
> python Automata walker
```

## The Sierpinski class
Uses the 'sierpinski' parameters to simulate a Sierpinski triangle.

```bash
> python Automata sierpinski
```

## The Sandpile class
Uses the 'sandpile' parameters to simulate an Abelian sandpile.

```bash
> python Automata sandpile
```

## The AutoCell class
> raise NotImplementedError("These docs are still to be fleshed out!")

Uses the 'autocell' parameters to simulate cell based automata. Implemented for:
- 1D (x) evolution into 2<sup>nd</sup> (spatial) dimension (y), or
- 2D (x-y) evolution into 3<sup>rd</sup> (temporal) dimension (t).

```bash
python Automata autocell
```

---
## References

### Markdown
- Preview in VSCode: <kbd>Ctrl</kbd> + <kbd>K</kbd> then <kbd>V</kbd>
- [Markdown cheat sheet](https://www.markdownguide.org/cheat-sheet/)

### Logging information:
- [Realpython logging](https://realpython.com/python-logging/)
- [Documentation: logging](https://docs.python.org/3/library/logging.html)

### Color maps:
- [matplotlib color maps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
- [Online cosine gradient generator](http://dev.thi.ng/gradients/)
    - Used as `palette(x) = a + b * cos(2π(c * t + d))`

### Gaussian filtering:
- [Documentation: scipy.ndimage.gaussian_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)
    - `from scipy.ndimage import gaussian_filter as gauss_smooth`

### Finding line segment intersection:
> DEPRECATED from Canvas for folding from 3x3 to 1x1 canvas (Canvas.foldCanvas()), more efficient at cost of 9x larger canvas
- [Wikipedia: Intersection](https://en.wikipedia.org/wiki/Intersection_(geometry)#Two_line_segments)
- [Documentation: numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- [Wikipedia: Cramer's rule](https://en.wikipedia.org/wiki/Cramer%27s_rule)

### Abelian sandpiles
- [Wikipedia: Abelian sandpile model](https://en.wikipedia.org/wiki/Abelian_sandpile_model)
- Lang, M., & Shkolnikov, M. (2019). [Harmonic dynamics of the abelian sandpile](https://www.pnas.org/doi/full/10.1073/pnas.1812015116). Proceedings of the National Academy of Sciences, 116(8), 2821-2830.
- Creutz, M. (1991). [Abelian sandpiles](https://pubs.aip.org/aip/cip/article-abstract/5/2/198/136705/Abelian-sandpiles?redirectedFrom=fulltext). Computers in physics, 5(2), 198-203.

---
## TODO: Next Steps
- [x] Make TODO list
- Automata
    - [ ] define `python Automata help|-help|-h` to return help docs in the command line
- Canvas
    - [ ] Gaussian filtering for smoothing
    - [ ] multiprocessing should be possible as walkers independent of one another
        - _I.E._ N canvases parsing total_walkers / N walkers each
- Walker
    - [ ] Find out if choice has more efficient function
    - [ ] Add Walker vision implementation to allow directional preference
        - _I.E._ Fix Walker.look() method, especially handling and 'type' of vis / self.vis
        - _I.E._ vision: [int, int] | np.array([int, int]).astype(int)
- Sandpile
    - [ ] Find out how large history should be (t-axis)
        - _I.E._ dynamically calculate t if possible
    - [ ] Sandpile.animate() save with PIL for proper .gif format
    - [ ] Test Sandpile.show3D() for better plots
- Bezier
    - [ ] Add Bezier.py to folder structure
    - [ ] Implement Bezier() in the \_\_main__.py file
    - [ ] Add docs to README above
- AutoCell
    - [x] Add original Autocell.py to folder structure
    - [ ] Implement AutoCell() in the \_\_main__.py file
    - [ ] Finish correcting typing in AutoCell.py
        - _I.E._ Includes parsing parameters from dictionary
    - [ ] Add docs to README above
    - [ ] Add AutoCell References
    - [ ] Cell1D show revamp
    - [ ] Checkbox for black/white in show 2D
    - [ ] Cell2D.animate() not called properly when run in Terminal
    - [ ] Cell2D resize not working
    - [ ] Have way to show age
        - _I.E._ Pixel gets darker the older it is