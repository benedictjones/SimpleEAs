# SimpleEAs
Python library for some simple Evolutionary Algorithms.
These follow an ask/tell work flow where one:
- new trial members/sample are requested using ask(), these need to me manually evaluated
- the fitness socores are then fed back to the object with tell() to enable a population update

## Algorithms
Algorithms to me implemented:
- Differential Evolution (DE)
- OpenAI ES (OAIES)


### DE
An example of DE solving a 5th order polynomial to fit noisy cos(x) data.

![](example_DE.gif)
