# 2FeldTherm
A two-feldspar thermometer based on the approach of Furhmam and Lindsley (1988) and Elkins and Grove (1990).
The main advantage of this approach is that it calculates three independent, and therefore redundant, temperatures based on the three feldspar components – albite, orthoclase and anorthite. This provides a means to assess the equilibrium between the two feldspars. If the three calculated temperatures diverge substantially, then the feldspar pair is likely not in equilibrium.

This package also contains a perturbation approach which adjusts both feldspar compositions by a fraction of a mol% and finds the best fitting temperatures. The steep gradient of the ternary-feldspar solvus means that a small change in composition can result in a large change in calculated temperature. Unfortunately, analysing feldspar compositions is often complicated by Na-loss or migration in response to an electron beam. This approach avoids the assumption that the measured compositions are exact. 

Example usage of the thermometer can be found in the *2FeldTherm.ipynb* notebook, including both the regular and perturbed approaches on individual and multiple feldspar pairs.

**Interaction Parameters**

The thermometer relies on ternary feldspar interaction, or mixing, parameters. This package contains several different sets from the literature, which are stored in their own class:

```python
import ternary_feldspar_therm as tf
tf.interaction_parameters.display_all()

## Available Interaction Parameters
## G1984: Ghiorso 1984
## GU1986: Green and Usdansky 1986
## NB1987: Nekvasil and Burnham 1987
## LN1988: Lindsley and Nekvasil 1988
## FL1988: Fuhrman and Lindsley 1988
## EG1990: Elkins and Grove 1990
## B2004_Al: Benisek et al 2004 Aluminium Avoidance
## B2004_MM: Benisek et al 2004 Molecular Mixing
```

Additional parameters can be added easily as a dataframe.

**References and Further Reading**

* Benisek, A., Kroll, H., & Cemič, L. (2004). New developments in two-feldspar thermometry. American Mineralogist, 89(10), 1496–1504. https://doi.org/10.2138/am-2004-1018
* Elkins, L. T., & Grove, T. L. (1990). Ternary feldspar experiments and thermodynamic models. American Mineralogist, 75(5–6), 544–559. https://pubs.geoscienceworld.org/msa/ammin/article-abstract/75/5-6/544/42371/Ternary-feldspar-experiments-and-thermodynamic
* Fuhrman, M. L., & Lindsley, D. H. (1988). Ternary-feldspar modeling and thermometry. American Mineralogist, 73(3–4), 201–215. https://pubs.geoscienceworld.org/msa/ammin/article-abstract/73/3-4/201/42101/Ternary-feldspar-modeling-and-thermometry
* Ghiorso, M. S. (1984). Activity/composition relations in the ternary feldspars. Contributions to Mineralogy and Petrology, 87(3), 282–296. https://doi.org/10.1007/BF00373061
* Green, N. L., & Usdansky, S. I. (1986). Ternary-feldspar mixing relations and thermobarometry. American Mineralogist, 71(9–10), 1100–1108. https://pubs.geoscienceworld.org/msa/ammin/article-abstract/71/9-10/1100/41934/Ternary-feldspar-mixing-relations-and
* Nekvasil, H., & Burnham, C. (1987). The calculated individual effects of pressure and water content on phase equilibria in the granite system. In Magmatic processes: Physicochemical principles; a volume in honor of Hatten S. Yoder, jr. The Geochemical Society. https://www.geochemsoc.org/publications/sps/v1magmaticprocesses
* Wen, S., & Nekvasil, H. (1994). SOLVALC: An interactive graphics program package for calculating the ternary feldspar solvus and for two-feldspar geothermometry. Computers & Geosciences, 20(6), 1025–1040. https://doi.org/10.1016/0098-3004(94)90039-6
