Venice is a code coupling scheme based on a tree structure that is generated using the connected components of a coupling timescale graph. It is inspired by the Hamiltonian splitting method of adaptive timesteps in gravitational dynamics. This implementation is based on the AMUSE framework, and supports particle and grid based data.

The name Venice is inspired by the image of all codes being little islands connected by Bridges. The Bridge scheme of coupling dynamical codes served as an important inspiration for this scheme. 

A Venice coupled system is set up in a few steps:
- Adding the codes to be coupled.
- Defining connections between data sets of codes.
- Defining coupling timescales between codes.

Those three elements define a functional system, but it has a couple of additional features:
- Kick functions can let one code impact another (the classical Bridge interaction is an example of this).
- Data sets can be manipulated on the fly; if one code generates or deletes particles, this can be communicated to another code.
- Coupling timesteps can be changed dynamically during evolution. 

The figures in the Venice paper can be reproduced using the paper\_{model}.py scripts; the generate\_paper\_figures.csh script runs them all in the correct order. Note that by default its files (which are a lot) are written within the repository's directory.
